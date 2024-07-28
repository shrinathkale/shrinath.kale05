class Library:
    def __init__(self):
        self.books = []
        self.noBooks = 0
    def addbook(self,book):
        self.books.append(book)
        self.noBooks = len(self.books)
    def show(self):
        print(f"Library contains {self.noBooks} books")
        print("The books in the library are:")
        for book in self.books:
            print(book)
obj = Library()
obj.addbook("OOP")
obj.addbook("COLD")
obj.addbook("DSA")
obj.addbook("DM")
obj.addbook("BCN")
obj.addbook("Motivation - THE ENERGY")
obj.show()