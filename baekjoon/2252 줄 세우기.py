import sys

check = []

lines = sys.stdin.readlines()
[vertex, line] = list(map(int, lines[0].split()))

check_dict = {}
for j in range(1, vertex+1):
    check_dict[j] = []

for i in range(1, len(lines)):
    [a, b] = list(map(int, lines[i].split()))
    check_dict[a].append(b)

print(check_dict)

stack = [1]
sorted = []
visited = []

while len(sorted) < vertex:
    if len(stack) > 0:
        student = stack[-1]
        visited.append(student)

    else:
        for l in range(1, vertex+1):
            if l not in visited:
                stack.append(l)
                break

        student = stack[-1]
        visited.append(student)

    if len(check_dict[student]) != 0:
        for k in range(len(check_dict[student])):
            item = check_dict[student].pop(-1)
            if item not in visited:
                stack.append(item)

    else:
        stack.remove(student)
        sorted.insert(0, student)

for k in range(len(sorted)):
    print(sorted[k], end = " ")
