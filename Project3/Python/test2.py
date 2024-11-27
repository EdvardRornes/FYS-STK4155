class Test:

    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def create_data(self):
        self.data = vars(self)
        return self.data


test = Test(1,2)
data = test.create_data()



class Parent:

    def __init__(self) -> None:
        pass

    def method1(self):
        print(f"{vars(self)}")

class child(Parent):

    def __init__(self, x):
        self.x = x 
