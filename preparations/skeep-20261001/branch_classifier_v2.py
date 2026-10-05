"""Frozen, offline, blind secondary first-action measurement.

This module is a bounded measurement grammar, never a shell interpreter. It
reads blind tool functions only and makes no command, provider or tool calls.
See FIRST_ACTION_V2_CODEBOOK.md for the exhaustive combination policy.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import importlib.util
import json
from pathlib import Path
import re
import sys

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('skeep_first_action_v1_helpers', HERE / 'branch_analysis.py')
v1 = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = v1
_spec.loader.exec_module(v1)
AgentRuntime = v1.AgentRuntime

RULE_VERSION = 'skeep-first-action/blind-secondary-v2.1'
RULE_FREEZE_UTC = '2026-10-03T05:07:46Z'
CATEGORIES = ('inspect_only', 'work', 'execution', 'poll', 'cancel', 'evidence_read', 'no_tool', 'unknown')
MAX_COMMAND_CHARS = 262144
MAX_TOKENS = 8192
MAX_COMPONENTS = 512
MAX_HEREDOCS = 32
MEASUREMENT_RULES = {
    'rule_version': RULE_VERSION,
    'rule_freeze_utc': RULE_FREEZE_UTC,
    'first_action': 'first original tool-list element; native AgentRuntime argument decoder',
    'grammar_failure': 'unknown; no substring salvage of recognized writes',
    'valid_compound_precedence': ['work', 'unknown', 'execution', 'inspect_only'],
    'write_evidence': 'explicit mutation utility/flag or non-null output file redirection',
    'execution': 'program/script/build/test/content transformation; opaque program bodies',
    'metadata': 'recognized read/selection/count/help/version/echo/date utilities',
    'detached_inspection': 'execution: detached_launch_not_inspect_only',
    'paths': 'lexical only; no cwd, glob, environment or filesystem inference',
    'model_calls': 0,
}


@dataclass(frozen=True)
class Token:
    kind: str
    value: str
    start: int
    end: int
    glob: bool = False
    quoted: bool = False


@dataclass
class Component:
    words: list[Token]
    redirects: list[tuple[str, Token]]
    heredoc: bool = False


class GrammarError(ValueError):
    pass


OPERATORS = (';;&', '&>>', '<<<', '<<-', '&&', '||', '>>', '<<', '<&', '>&',
             '&>', '>|', '<>', '|&', ';;', ';&', ';', '|', '&', '>', '<', '(', ')')
SEPARATORS = {';', '&&', '||', '|'}
REDIRECTS = {'>', '>>', '>|', '<', '<>', '<&', '>&', '&>', '&>>', '<<', '<<-'}
MUTATORS = v1.WRITE - {'tee'}
EXECUTORS = (v1.COMPUTE - {'wc', 'echo', 'printf'}) | v1.SCRIPTS | {
    'awk', 'gawk', 'mawk', 'timeout', 'sleep', 'xargs', 'cargo', 'go', 'npm',
    'npx', 'yarn', 'pnpm', 'uv', 'pip', 'pip3', 'javac', 'java', 'dotnet',
    'iverilog', 'verilator', 'vvp', 'yosys', 'zip', 'unzip', 'tar', 'gzip', 'gunzip',
}
METADATA_PROGRAMS = EXECUTORS | MUTATORS | v1.INSPECT | {
    'wc', 'tee', 'md5sum', 'sha1sum', 'sha224sum', 'sha256sum', 'sha384sum',
    'sha512sum', 'nm', 'nproc', 'free', 'ps', 'find',
}


def _lex(command: str) -> list[Token]:
    """Quote-aware tokens; consume heredoc bodies without tokenizing them."""
    if not isinstance(command, str) or not command.strip() or '\x00' in command:
        raise GrammarError('missing_or_invalid_command')
    if len(command) > MAX_COMMAND_CHARS:
        raise GrammarError('command_bound_exceeded')
    result, pending = [], []
    i = 0
    heredoc_count = 0
    expect_delimiter = None

    def append(token):
        result.append(token)
        if len(result) > MAX_TOKENS:
            raise GrammarError('token_bound_exceeded')

    while i < len(command):
        char = command[i]
        if char in ' \t\r':
            i += 1
            continue
        if char == '\n':
            if expect_delimiter:
                raise GrammarError('invalid_heredoc_delimiter')
            append(Token('op', '\n', i, i + 1))
            i += 1
            for delimiter, strip_tabs, quoted in pending:
                found = False
                body = []
                while i < len(command):
                    end = command.find('\n', i)
                    end = len(command) if end < 0 else end
                    line = command[i:end]
                    candidate = line.lstrip('\t') if strip_tabs else line
                    i = end + (end < len(command))
                    if candidate == delimiter:
                        found = True
                        break
                    body.append(line)
                if not found:
                    raise GrammarError('unterminated_heredoc')
                # Unquoted delimiters cause Bash expansion before stdin delivery.
                # Never reinterpret those body expansions as top-level commands.
                if not quoted and any('$' in line or '`' in line for line in body):
                    raise GrammarError('unsupported_heredoc_expansion')
            pending = []
            continue
        if char == '#':
            end = command.find('\n', i)
            i = len(command) if end < 0 else end
            continue
        if char == '\\' and i + 1 < len(command) and command[i + 1] == '\n':
            i += 2
            continue
        operator = next((op for op in OPERATORS if command.startswith(op, i)), None)
        if operator:
            if expect_delimiter:
                raise GrammarError('invalid_heredoc_delimiter')
            append(Token('op', operator, i, i + len(operator)))
            i += len(operator)
            if operator in {'<<', '<<-'}:
                expect_delimiter = operator
            continue
        start, chars, glob, quoted = i, [], False, False
        while i < len(command):
            char = command[i]
            if char in ' \t\r\n' or any(command.startswith(op, i) for op in OPERATORS):
                break
            if char in "'\"":
                quote = char
                quoted = True
                i += 1
                while i < len(command) and command[i] != quote:
                    char = command[i]
                    if quote == '"' and (char == '`' or (char == '$' and i + 1 < len(command)
                            and re.match(r'[A-Za-z0-9_@*#?$!{(]', command[i + 1]))):
                        raise GrammarError('unsupported_dynamic_expansion')
                    if quote == '"' and char == '\\' and i + 1 < len(command):
                        following = command[i + 1]
                        if following == '\n':
                            i += 2
                            continue
                        if following in '\\$`"':
                            chars.append(following)
                            i += 2
                            continue
                    chars.append(char)
                    i += 1
                if i == len(command):
                    raise GrammarError('unclosed_quote')
                i += 1
            elif char == '\\':
                if i + 1 == len(command):
                    raise GrammarError('trailing_escape')
                if command[i + 1] != '\n':
                    quoted = True
                    chars.append(command[i + 1])
                i += 2
            elif char in '$`' or char in '{}' or (char == '~' and not chars):
                raise GrammarError('unsupported_dynamic_expansion')
            else:
                glob |= char in '*?[]'
                chars.append(char)
                i += 1
        token = Token('word', ''.join(chars), start, i, glob, quoted)
        append(token)
        if expect_delimiter:
            if not token.value or token.glob:
                raise GrammarError('invalid_heredoc_delimiter')
            heredoc_count += 1
            if heredoc_count > MAX_HEREDOCS:
                raise GrammarError('heredoc_bound_exceeded')
            pending.append((token.value, expect_delimiter == '<<-', token.quoted))
            expect_delimiter = None
    if expect_delimiter or pending:
        raise GrammarError('unterminated_heredoc')
    return result


def _parse(command: str) -> list[Component]:
    tokens = _lex(command)
    components, current = [], Component([], [])
    last_separator = None
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token.kind == 'word':
            current.words.append(token)
            last_separator = None
        elif token.value in REDIRECTS:
            if i + 1 == len(tokens) or tokens[i + 1].kind != 'word':
                raise GrammarError('invalid_redirection')
            descriptor = None
            if current.words and current.words[-1].value.isdecimal() and not current.words[-1].quoted and current.words[-1].end == token.start:
                descriptor = current.words.pop().value  # Adjacent IO number; a spaced numeric operand stays an operand.
            target = tokens[i + 1]
            if token.value == '>&' and descriptor is not None and not (target.value.isdecimal() or target.value == '-'):
                raise GrammarError('unsupported_output_descriptor_target')
            if token.value == '<&' and not (target.value.isdecimal() or target.value == '-'):
                raise GrammarError('unsupported_input_descriptor_target')
            if token.value in {'<<', '<<-'}:
                current.heredoc = True
            else:
                current.redirects.append((token.value, target))
            i += 1
            last_separator = None
        elif token.value in SEPARATORS or token.value == '\n':
            if token.value == '\n' and not current.words and not current.redirects and not current.heredoc:
                i += 1
                continue
            if current.words:
                components.append(current)
                if len(components) > MAX_COMPONENTS:
                    raise GrammarError('component_bound_exceeded')
                current = Component([], [])
            elif current.redirects or current.heredoc:
                raise GrammarError('redirect_only_component')
            else:
                raise GrammarError('empty_compound_component')
            last_separator = ';' if token.value == '\n' else token.value
        else:
            raise GrammarError('unsupported_shell_operator')
        i += 1
    if current.words:
        components.append(current)
    elif current.redirects or current.heredoc:
        raise GrammarError('redirect_only_component')
    elif last_separator in {'&&', '||', '|'}:
        raise GrammarError('incomplete_compound_command')
    if not components:
        raise GrammarError('missing_or_invalid_command')
    if len(components) > MAX_COMPONENTS:
        raise GrammarError('component_bound_exceeded')
    if any(c.words[0].value in v1.RESERVED or '=' in c.words[0].value for c in components):
        raise GrammarError('unsupported_shell_construct')
    return components


def _result(category, reason, files=None):
    return category, reason, files


def _sed_inplace_option(args):
    """Read flags as flags; expression/file option values stay opaque."""
    i = 0
    while i < len(args):
        word = args[i]
        if word == '--':
            return False
        if word == '--in-place' or word.startswith('--in-place='):
            return True
        if word in {'--expression', '--file'}:
            i += 2
            continue
        if word.startswith('--expression=') or word.startswith('--file='):
            i += 1
            continue
        if word.startswith('--'):
            if word not in {'--quiet', '--silent', '--regexp-extended', '--separate',
                            '--unbuffered', '--null-data', '--posix', '--sandbox', '--debug'}:
                return False
        elif word.startswith('-') and word != '-':
            cluster, offset = word[1:], 0
            while offset < len(cluster):
                flag = cluster[offset]
                if flag == 'i':
                    return True
                if flag in {'e', 'f'}:
                    if offset + 1 == len(cluster):
                        i += 1  # Following word is an opaque script/file value.
                    break
                if flag not in 'nErsuz':
                    return False
                offset += 1
        i += 1
    return False


def _printf_assignment(args):
    """Recognize builtin -vNAME and actual %n conversion, not %%n text."""
    if not args:
        return False
    if args[0].startswith('-v'):
        return True
    offset = 1 if args[0] == '--' else 0
    if offset == len(args):
        return False
    fmt, i = args[offset], 0
    while i < len(fmt):
        if fmt[i] != '%':
            i += 1
            continue
        i += 1
        if i < len(fmt) and fmt[i] == '%':
            i += 1
            continue
        # Bash %(strftime-format)T has a separate format language. Its %n is
        # a newline, never the printf variable assignment directive.
        if i < len(fmt) and fmt[i] == '(':
            end = fmt.find(')', i + 1)
            if end >= 0 and end + 1 < len(fmt) and fmt[end + 1] == 'T':
                i = end + 2
                continue
        while i < len(fmt) and fmt[i] in '-+ #0':
            i += 1
        while i < len(fmt) and (fmt[i].isdigit() or fmt[i] == '*'):
            i += 1
        if i < len(fmt) and fmt[i] == '.':
            i += 1
            while i < len(fmt) and (fmt[i].isdigit() or fmt[i] == '*'):
                i += 1
        while i < len(fmt) and fmt[i] in 'hljztL':
            i += 1
        if i < len(fmt) and fmt[i] == 'n':
            return True
        i += 1
    return False


def _component(component: Component):
    words = [w.value for w in component.words]
    name, args = v1._utility(words[0]), words[1:]
    if component.words[0].glob:
        return _result('unknown', 'dynamic_command_name')
    # Metadata exception is limited to known utilities/programs and an exact
    # one-flag shape. An arbitrary ./program --help remains an opaque execution.
    if name in METADATA_PROGRAMS and args in (['--help'], ['--version']):
        return _result('inspect_only', 'recognized_help_or_version', [])
    if name in {'python', 'python2', 'python3'} and args in (['-V'], ['-h']):
        return _result('inspect_only', 'recognized_help_or_version', [])
    if name == 'sed' and _sed_inplace_option(args):
        return _result('work', 'recognized_inplace_edit')
    if name in v1.INSPECT:
        files = v1._inspection(name, args)
        if files is not None:
            return _result('inspect_only', 'recognized_inspection', files)
        if name == 'sed':
            parsed = v1._options(args, short='nEr', value_short='ef',
                                 long=('--quiet', '--silent', '--regexp-extended'),
                                 value_long=('--expression', '--file'))
            if parsed and (parsed[0] or parsed[1]):
                return _result('execution', 'opaque_sed_script')
        return _result('unknown', 'unsupported_inspection_options_or_script')
    if name == 'cd':
        parsed = v1._options(args, short='LP', long=())
        if parsed and len(parsed[0]) == 1 and parsed[0][0] != '-':
            return _result('inspect_only', 'recognized_directory_selection')
        return _result('unknown', 'unsupported_cd')
    if name in {'echo', 'printf'}:
        if name == 'printf' and _printf_assignment(args):
            return _result('work', 'recognized_shell_variable_assignment')
        return _result('inspect_only', 'recognized_literal_output', [])
    if name == 'date':
        if any(a == '-s' or a == '--set' or a.startswith('--set=') for a in args):
            return _result('work', 'recognized_date_setting')
        if all(a in {'-u', '--utc', '--universal', '-R', '--rfc-email', '-I', '--iso-8601',
                     '--help', '--version'} or a.startswith('+') or a.startswith('--iso-8601=') for a in args):
            return _result('inspect_only', 'recognized_clock_metadata', [])
        return _result('unknown', 'unsupported_date_options')
    if name == 'wc':
        parsed = v1._options(args, short='clmwL', long=(
            '--bytes', '--chars', '--lines', '--words', '--max-line-length'))
        return _result('inspect_only', 'recognized_count_metadata', parsed[0]) if parsed else _result(
            'unknown', 'unsupported_wc_options')
    if name in {'which', 'type'}:
        parsed = v1._options(args, short='as' if name == 'which' else 'afptP', long=())
        return _result('inspect_only', 'recognized_program_metadata', []) if parsed and parsed[0] else _result(
            'unknown', 'unsupported_program_metadata_options')
    if name == 'command' and args and args[0] in {'-v', '-V'} and len(args) > 1:
        return _result('inspect_only', 'recognized_program_metadata', [])
    if name in {'md5sum', 'sha1sum', 'sha224sum', 'sha256sum', 'sha384sum', 'sha512sum'}:
        parsed = v1._options(args, short='btczw', long=(
            '--binary', '--text', '--check', '--zero', '--warn', '--quiet', '--status', '--strict', '--tag'))
        return _result('inspect_only', 'recognized_checksum_metadata', parsed[0]) if parsed else _result(
            'unknown', 'unsupported_checksum_options')
    if name == 'nm':
        parsed = v1._options(args, short='aACDglnoprSu', long=(
            '--debug-syms', '--print-file-name', '--demangle', '--dynamic', '--extern-only',
            '--line-numbers', '--numeric-sort', '--no-sort', '--print-size', '--undefined-only'))
        return _result('inspect_only', 'recognized_symbol_metadata', parsed[0]) if parsed else _result(
            'unknown', 'unsupported_nm_options')
    if name == 'nproc':
        parsed = v1._options(args, long=('--all',), value_long=('--ignore',))
        if parsed and not parsed[0] and all(v.isdecimal() for _, v in parsed[1]):
            return _result('inspect_only', 'recognized_system_metadata', [])
        return _result('unknown', 'unsupported_nproc_options')
    if name == 'free':
        parsed = v1._options(args, short='bkmgtwhl', long=(
            '--bytes', '--kibi', '--mebi', '--gibi', '--tebi', '--wide', '--human', '--lohi', '--total'))
        return _result('inspect_only', 'recognized_system_metadata', []) if parsed and not parsed[0] else _result(
            'unknown', 'unsupported_free_options')
    if name == 'ps':
        if all(a in {'aux', 'ax', 'u', '-e', '-f', '-ef', '-A', '-a', '-x'} for a in args):
            return _result('inspect_only', 'recognized_process_metadata', [])
        return _result('unknown', 'unsupported_ps_options')
    if name in {'true', 'false'} and not args:
        return _result('inspect_only', 'recognized_status_utility', [])
    if name == 'find':
        # Only a bounded path/predicate subset; -exec/-delete/-fprint and every
        # other action or option are unknown. Pattern operands are not paths.
        files, offset = [], 0
        while offset < len(args) and not args[offset].startswith('-') and args[offset] != '!':
            files.append(args[offset])
            offset += 1
        if not files:
            return _result('unknown', 'unsupported_find_expression')
        while offset < len(args):
            flag = args[offset]
            if flag in {'-o', '-or', '-a', '-and', '!', '-not', '-print', '-print0'}:
                offset += 1
                continue
            if flag not in {'-maxdepth', '-mindepth', '-type', '-name', '-iname', '-path', '-ipath'} or offset + 1 == len(args):
                return _result('unknown', 'unsupported_find_expression')
            value = args[offset + 1]
            if flag in {'-maxdepth', '-mindepth'} and not value.isdecimal():
                return _result('unknown', 'unsupported_find_expression')
            if flag == '-type' and value not in {'b', 'c', 'd', 'f', 'l', 'p', 's'}:
                return _result('unknown', 'unsupported_find_expression')
            offset += 2
        return _result('inspect_only', 'recognized_file_search', files)
    if name in MUTATORS:
        return _result('work', 'recognized_mutation_utility', [a for a in args if not a.startswith('-')])
    if name == 'tee':
        parsed = v1._options(args, short='ai', long=('--append', '--ignore-interrupts'))
        if not parsed:
            return _result('unknown', 'unsupported_tee_options')
        return _result('work', 'recognized_mutation_utility', parsed[0]) if any(a != '-' for a in parsed[0]) else _result(
            'execution', 'recognized_content_transformation', [])
    if name in EXECUTORS:
        reason = 'opaque_program_or_script' if name in v1.SCRIPTS or name in {'awk', 'gawk', 'mawk'} else 'recognized_program_execution'
        return _result('execution', reason)
    if '/' in words[0]:
        return _result('execution', 'opaque_executable_path', [words[0]])
    return _result('unknown', 'unsupported_command')


def classify_shell(command):
    try:
        parsed = _parse(command)
    except GrammarError as exc:
        return dict(category='unknown', reason=str(exc), static_paths=None)
    components = []
    files = []
    paths_complete = True
    for component in parsed:
        category, reason, paths = _component(component)
        redirects = []
        for op, target in component.redirects:
            descriptor = op in {'>&', '<&'} and (target.value.isdecimal() or target.value == '-')
            sink = target.value == '/dev/null' and not target.glob
            output = op in {'>', '>>', '>|', '<>', '&>', '&>>', '>&'}
            if output and not descriptor and not sink:
                category, reason = 'work', 'recognized_output_file_redirection'
            redirects.append(dict(operator=op, kind='descriptor' if descriptor else 'null_sink' if sink else 'file'))
            if not descriptor and not sink:
                files.append(target.value)
                paths_complete &= not target.glob
        components.append(dict(category=category, reason=reason, redirects=redirects))
        if paths is None:
            paths_complete = False
        else:
            files.extend(paths)
        # Globs are accepted for classification, never expanded for path evidence.
        if any(w.glob for w in component.words):
            paths_complete = False
    category = next(c for c in ('work', 'unknown', 'execution', 'inspect_only')
                    if any(row['category'] == c for row in components))
    reason = 'all_components_inspection' if category == 'inspect_only' else next(
        row['reason'] for row in components if row['category'] == category)
    return dict(category=category, reason=reason,
                static_paths=v1._paths(files) if paths_complete else None, components=components)


def classify_action(assistant):
    """First original tool only; no normalized/reordered tool or body inference."""
    if not isinstance(assistant, dict) or assistant.get('role') != 'assistant':
        return dict(category='unknown', reason='invalid_assistant_record', static_paths=None)
    calls = assistant.get('tool_calls')
    if calls is None or calls == []:
        return dict(category='no_tool', reason='accepted_no_tool_response', static_paths=None)
    if not isinstance(calls, list) or not calls or not isinstance(calls[0], dict):
        return dict(category='unknown', reason='invalid_tool_list', static_paths=None)
    call = calls[0]
    function = call.get('function')
    if not isinstance(function, dict) or not isinstance(function.get('name'), str):
        return dict(category='unknown', reason='invalid_first_action', static_paths=None)
    name = function['name']
    try:
        args = AgentRuntime._tool_arguments(call)
    except (ValueError, TypeError, KeyError):
        return dict(category='unknown', reason='unparseable_native_arguments', tool=name, static_paths=None)
    if name in {'bash', 'bash_start'}:
        classified = classify_shell(args.get('command'))
        if name == 'bash_start' and classified['category'] == 'inspect_only':
            classified.update(category='execution', reason='detached_launch_not_inspect_only')
        return dict(tool=name, **classified)
    separate = {'bash_poll': 'poll', 'bash_cancel': 'cancel', 'read_tool_evidence': 'evidence_read'}
    return dict(category=separate.get(name, 'unknown'), tool=name, static_paths=None,
                reason='separate_native_action' if name in separate else 'unsupported_tool')


def classify_blind(corpus):
    """Accept the blind schema only; output no source/group/model information."""
    if not isinstance(corpus, dict) or corpus.get('schema') != 'skeep-blind-first-actions/v1':
        raise ValueError('expected blind first-action corpus')
    actions = corpus.get('actions')
    if not isinstance(actions, list):
        raise ValueError('expected blind actions list')
    rows, seen = [], set()
    for action in actions:
        if not isinstance(action, dict) or set(action) != {'blind_id', 'assistant'}:
            raise ValueError('expected blind_id and assistant only')
        blind_id = action['blind_id']
        if not isinstance(blind_id, str) or not re.fullmatch(r'a\d{4}', blind_id) or blind_id in seen:
            raise ValueError('invalid or duplicate blind_id')
        seen.add(blind_id)
        rows.append(dict(blind_id=blind_id, **classify_action(action['assistant'])))
    categories = Counter(row['category'] for row in rows)
    reasons = Counter(row['reason'] for row in rows)
    return dict(schema='skeep-blind-first-action-classifications/v2', rule_version=RULE_VERSION,
                measurement_rules=MEASUREMENT_RULES, action_count=len(rows),
                category_counts={c: categories[c] for c in CATEGORIES},
                reason_counts=dict(sorted(reasons.items())), classifications=rows, model_calls=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blind-actions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = classify_blind(json.loads(args.blind_actions.read_text()))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('rule_version', 'action_count', 'category_counts', 'reason_counts', 'model_calls')}))


if __name__ == '__main__':
    main()
