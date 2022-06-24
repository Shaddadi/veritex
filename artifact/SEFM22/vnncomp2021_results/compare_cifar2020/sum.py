'stanley bak'

def main():
    'main entry point'

    vio2021 = 0
    holds2021 = 0

    with open('2021.csv') as f:
        for line in f:
            if '(v)' in line:
                vio2021 += 1
            elif '(h)' in line:
                holds2021 += 1

    unknown2021 = 138 - vio2021 - holds2021
    print(f"2021, violated: {vio2021}, holds: {holds2021}, unknown: {unknown2021}")

    sat_indices = set()
    unsat_indices = set()
    unknown_indices = set(range(1, 139))

    for findex in range(1, 7):
        with open(f'{findex}.txt') as f:
            line_num = 0
            for line in f:
                line = line.strip().lower()
                line_num += 1

                if line_num > 138:
                    break

                if "unsat" in line:
                    unsat_indices.add(line_num)

                    if line_num in unknown_indices:
                        unknown_indices.remove(line_num)
                elif "sat" in line:
                    sat_indices.add(line_num)

                    if line_num in unknown_indices:
                        unknown_indices.remove(line_num)

    print(f"2020, violated: {len(sat_indices)}, holds: {len(unsat_indices)}, unknown: {len(unknown_indices)}")

    assert len(unknown_indices) + len(sat_indices) + len(unsat_indices) == 138
    
if __name__ == "__main__":
    main()
