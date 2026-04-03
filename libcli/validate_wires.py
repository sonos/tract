#!/usr/bin/env python3
"""Validate wire drawing continuity in tract dump output.

Top-down approach: track which wires are active at each visual column,
update state as we encounter node lines, swap lines, and split lines.
Flag any discontinuity we can't explain.
"""

import sys
import re

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def parse_colored_chars(line):
    """Return list of (char, ansi_color_code) for each visible character."""
    result = []
    current_color = ""
    i = 0
    while i < len(line):
        if line[i] == '\x1b':
            j = line.index('m', i) + 1
            seq = line[i:j]
            if seq == '\x1b[0m':
                current_color = ""
            else:
                current_color = seq
            i = j
        else:
            result.append((line[i], current_color))
            i += 1
    return result

# Character classification
CONNECTS_DOWN = set('┃┣┏╋┳┓')
CONNECTS_UP   = set('┃┣╋┻┗╹')
CONNECTS_RIGHT = set('┣┏┗━╋┳┻')
CONNECTS_LEFT  = set('┓┛┗━╋┳┻')
ALL_WIRE = CONNECTS_DOWN | CONNECTS_UP | set('━')

def classify_line(wire_str):
    """Classify a wire-region string as: filler, node, swap, split, info, or unknown."""
    if not wire_str:
        return 'empty'
    chars = set(wire_str) - {' '}
    if not chars:
        return 'empty'
    # Info line: has ━━━ after spaces (output shape)
    if '━━━' in wire_str and ('┃' in wire_str or wire_str.strip().startswith('━')):
        only_vert_and_horiz = chars <= {'┃', '━', ' '}
        if only_vert_and_horiz:
            return 'info'
    # Filler: only vertical bars
    if chars <= {'┃'}:
        return 'filler'
    # Swap: has ┗ and ┓ (wire moves right)
    if '┗' in wire_str and '┓' in wire_str and '┣' not in wire_str:
        return 'swap'
    # Split: has ┣ and ┓ but no ┻ (wire clones right)
    if '┣' in wire_str and '┓' in wire_str and '┻' not in wire_str:
        return 'split'
    # Node: has ┣ with possible ┻ (inputs merging)
    if '┣' in wire_str and ('┻' in wire_str or '┓' in wire_str or wire_str.endswith('┣')):
        return 'node'
    # Source: starts with ┏
    if '┏' in wire_str:
        return 'node'
    # Single ┣ at end (1 input, 1 output)
    if wire_str.rstrip().endswith('┣'):
        return 'node'
    return 'unknown'

def extract_wire_region(raw_line):
    """Extract wire region from a line. Returns (offset, wire_string, colored_chars) or None."""
    stripped = strip_ansi(raw_line)
    colored = parse_colored_chars(raw_line)

    first = None
    for i, c in enumerate(stripped):
        if c in ALL_WIRE:
            first = i
            break
    if first is None:
        return None

    # Find end of wire region (stop at non-wire, non-space char)
    end = first
    for i in range(first, len(stripped)):
        if stripped[i] in ALL_WIRE:
            end = i + 1
        elif stripped[i] != ' ':
            break

    wire_str = stripped[first:end]
    wire_colored = colored[first:end] if first < len(colored) else []
    return (first, wire_str, wire_colored)

def find_node_id(stripped_line):
    """Try to extract node ID from a line like '... ┣┻ 123 OpName ...'"""
    m = re.search(r'[┃┣┻┓┗┏╋┳╹]\s+(\d+)\s+', stripped_line)
    if m:
        return int(m.group(1))
    return None

def main():
    raw_lines = [l.rstrip('\n') for l in sys.stdin.readlines()]

    errors = []
    prev_wire = None
    prev_colored = None
    prev_lineno = 0
    prev_kind = None
    prev_node_id = None

    for lineno_0, raw_line in enumerate(raw_lines):
        lineno = lineno_0 + 1
        stripped = strip_ansi(raw_line)
        r = extract_wire_region(raw_line)
        if r is None:
            continue

        offset, wire_str, wire_colored = r
        kind = classify_line(wire_str)
        node_id = find_node_id(stripped)

        if prev_wire is not None:
            prev_str = prev_wire
            # Check top-down: each column in prev that connects down
            # must have something valid below
            max_col = max(len(prev_str), len(wire_str))
            for col in range(max_col):
                pc = prev_str[col] if col < len(prev_str) else ' '
                cc = wire_str[col] if col < len(wire_str) else ' '

                if pc in CONNECTS_DOWN and cc == ' ':
                    # Wire going down disappears
                    # Acceptable if: current line is a node/filler line (wire consumed or output terminated)
                    if kind in ('node', 'filler', 'info'):
                        continue
                    errors.append((lineno, col, prev_lineno, prev_node_id, node_id,
                        f"col {col}: '{pc}' connects down into space"))

                elif pc in CONNECTS_DOWN and cc not in CONNECTS_UP and cc != '━' and cc not in CONNECTS_DOWN:
                    # Connecting down into something that doesn't connect up
                    # ┓ doesn't connect up — but it's OK in a swap/split context
                    if cc == '┓' and kind in ('swap', 'split'):
                        continue  # swap/split endpoint
                    errors.append((lineno, col, prev_lineno, prev_node_id, node_id,
                        f"col {col}: '{pc}' down into '{cc}' ({kind} line)"))

                elif cc in CONNECTS_UP and pc == ' ':
                    # Wire connecting up from nothing
                    # Acceptable if: this is a node line (new const/hidden input)
                    # or an info/filler line right after a source node
                    if kind in ('node', 'info', 'filler'):
                        continue
                    errors.append((lineno, col, prev_lineno, prev_node_id, node_id,
                        f"col {col}: '{cc}' up from space ({kind} line)"))

                elif cc in CONNECTS_UP and pc not in CONNECTS_DOWN and pc != '━':
                    # Connecting up from something that doesn't connect down
                    if pc == '┗' and kind in ('node', 'filler'):
                        continue  # ┗ was a swap, wire shifted
                    if pc == '┻' and kind == 'node':
                        continue  # stacked node inputs
                    errors.append((lineno, col, prev_lineno, prev_node_id, node_id,
                        f"col {col}: '{cc}' up from '{pc}' ({kind} line, prev={prev_kind})"))

                # Color continuity: ┃ to ┃ must be same color (not at node junction)
                if (pc == '┃' and cc == '┃'
                        and prev_colored is not None
                        and col < len(prev_colored) and col < len(wire_colored)):
                    pc_color = prev_colored[col][1]
                    cc_color = wire_colored[col][1]
                    if pc_color and cc_color and pc_color != cc_color:
                        errors.append((lineno, col, prev_lineno, prev_node_id, node_id,
                            f"col {col}: color change on ┃→┃ ({prev_kind}→{kind})"))

        prev_wire = wire_str
        prev_colored = wire_colored
        prev_lineno = lineno
        prev_kind = kind
        prev_node_id = node_id

    if not errors:
        print("All wire connections valid.")
        sys.exit(0)

    # Group by line pair and show in context
    shown = set()
    for lineno, col, prev_ln, prev_nid, curr_nid, msg in errors:
        key = (prev_ln, lineno)
        if key in shown:
            continue
        shown.add(key)
        errs_here = [(c, m) for l, c, pl, pn, cn, m in errors if pl == prev_ln and l == lineno]

        node_info = ""
        if prev_nid is not None:
            node_info += f" (after node {prev_nid})"
        if curr_nid is not None:
            node_info += f" (at node {curr_nid})"

        print(f"\n--- Lines {prev_ln}-{lineno}{node_info}: {len(errs_here)} error(s) ---")
        # Show context
        prev_r = extract_wire_region(raw_lines[prev_ln - 1])
        curr_r = extract_wire_region(raw_lines[lineno - 1])
        if prev_r:
            print(f"  {prev_r[1]}  [{classify_line(prev_r[1])}]")
        if curr_r:
            print(f"  {curr_r[1]}  [{classify_line(curr_r[1])}]")
        if prev_r and curr_r:
            marker = list(' ' * max(len(prev_r[1]), len(curr_r[1])))
            for c, m in errs_here:
                if c < len(marker):
                    marker[c] = '^'
            print(f"  {''.join(marker)}")
        for c, m in errs_here:
            print(f"    {m}")

    total = len(errors)
    print(f"\nTotal: {total} errors across {len(shown)} line pairs")
    sys.exit(1)

if __name__ == '__main__':
    main()
