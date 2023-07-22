
class Atm():

    def __init__(self):
        self.pin = ''
        self.balance = 0

        self.menu()

    def menu(self):
        user_input = input(
            '''
            Hello, How would you like to proceed?
            Enter 1 to create pin.
            Enter 2 to deposite.
            Enter 3 to withdraw.
            Enter 4 to check balance.
            Enter 5 to exit.
            '''
        )

        if user_input=='1':
            self.create_pin()
        elif user_input=='2':
            self.deposite()
        elif user_input=='3':
            self.withdraw()
        elif user_input=='4':
            self.check_balance()
        elif user_input=='5':
            self.exit()

    def create_pin(self):
        self.pin = input('Enter your pin: ')
        print("Pin created successfully")
        self.menu()
    
    def deposite(self):
        temp = input('Enter your pin: ')
        if temp == self.pin:
            self.balance = int(input("Enter your amount: "))
            print('Successfully deposited')
        else:
            print("Please enter correct pin")
        self.menu()
    
    def withdraw(self):
        temp = input('Enter your pin: ')
        if temp == self.pin:
            amount = int(input("Enter your amount: "))
            if amount < self.balance:
                self.balance -= amount
                print('please collect money.')
            else:
                print("please enter valid amount")
        else:
            print("Please enter correct pin")
        self.menu()
    
    def check_balance(self):
        temp = input("Enter your pin: ")
        if temp == self.pin:
            print(f"available balance: {self.balance}")
        else:
            print('please enter correct pin.')
        self.menu()
        
    def exit(self):
        print("Thank you, visit again!")

sbi = Atm()


        























