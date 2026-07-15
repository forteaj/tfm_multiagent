import string

def numbers():
    methods = []
    for i in range(10):
        # Fix: Use a default argument (index=i) to capture the current value of i. original: methods.append(lambda x: x + i)
        methods.append(lambda x, index=i: x + index)

    return methods


funcs = numbers()

print(funcs[0](10))
print(funcs[1](10))
print(funcs[2](10))