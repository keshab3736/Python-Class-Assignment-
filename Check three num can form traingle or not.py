def is_triangle(a, b, c):
    if a + b <= c or a + c <= b or b + c <= a:
        return False
    else:
        return True
        
side1 = float(input("Enter the first side: "))
side2 = float(input("Enter the second side: "))
side3 = float(input("Enter the third side: "))

if is_triangle(side1, side2, side3):
    print("Yes, the sides can form a triangle.")
else:
    print("No, the sides cannot form a triangle.")
