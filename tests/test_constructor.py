
class TestObject:

    Vrz = 0.5;

    def __init__(self, c=2, tau1=0.5):
        print(vars(self))
        print(vars())   # vars() is the same as locals() <- return objects in current namespace; however, vars(.) can also be used with arguments for other uses
        # Source: https://stackoverflow.com/questions/12191075/is-there-a-shortcut-for-self-somevariable-somevariable-in-a-python-class-con/12191118
        vars(self).update((k, v) for k, v in vars().items() if k != "self")

    def action1(self):
        return False

    @classmethod
    def action2(cls):
        return True

    @staticmethod
    def action3():
        return 10


if __name__ == '__main__':
    test_object = TestObject()
    print("-----")
    print(vars(test_object))
    print(dir(test_object))

