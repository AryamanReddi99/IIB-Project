def foo():
    return 1,2

a,b,c = *foo(), 3
print(a,b,c)