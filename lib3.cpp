#include<iostream>
#include<conio.h>
#include<windows.h>
#include<string>
using namespace std;
static int bookcount=0;
class BookArchive
{
	protected:
		int i,j;
		string bookcode[100];
	    char name[100][50];
	    char author[100][50];
	    int quantity[100], price[100];
	    int issue[100];
		
		BookArchive()
		{
			i=j=0;
		}
		
	public:
		void add_books();
		void list_of_books();
		void modify_books();
		void delete_books();
		void search_books();
		void issue_books();
		void return_books();
};
class Library : public BookArchive
{
	public:
		void introduction()
		{
			system("cls");
			cout<<"\n\n\n\t\t\t     WELCOME TO PROJECT\n";
			cout<<"\t\t\t    LIBRARY MANAGEMENT SYSTEM\n\n\n";
			cout<<"\t\t\tHere you can Issue,Return,Add \n" ;
			cout<<"\t\t\tMaintaining records of books.\n" ;
			cout<<"\t\t\tThis  Program  can  Hold Record\n\n" ;
			cout<<"\t\t    Let's Get Started :-)\n\n";
			cout<<"\t\t    Press any key to go to main menu...";
			getch();
		}
		void main_menu()
		{
			char ch;
			while (1)
			{
				system("cls");
				cout<<"\n\n\t\t\t\t MAIN  MENU\n";
				cout<<"\t\t\t\t------------------------\n\n";
				cout<<"\t\t\t1. Welcome Page\n\n";
				cout<<"\t\t\t2. View Book list\n\n";
				cout<<"\t\t\t3. Add Books\n\n";
				cout<<"\t\t\t4. Book Edit Menu\n\n";
				cout<<"\t\t\t5. Search Books\n\n";
				cout<<"\t\t\t6. Issue/Return Books\n\n";
				cout<<"\t\t\t0. Close Application\n\n\n";
				cout<<"\t\t\tEnter Your Choice : " ;
				cin>>ch;
				switch(ch)
				{
					case '1':
						introduction() ;
						break;
					case '2':
						list_of_books();
						break;
					case '3':
						add_books();
						break;
					case '4':
						edit_menu();
						break;
					case '5':
						search_books();
						break;
					case '6':
						issue_return();
					case '0':
					{
						system("cls");
						cout<<"\n\n\t\tThank You For Using the Program\n";
						cout<<"\t\tExiting in 3 second...........>\n\n\n";
						Sleep(3000);
						exit(0);
					}
					break;
					default:
					{
						cout<<"\t\t\t\a\aWrong Entry!!\n";
						cout<<"\t\t\tPlease Enter Correct Option";
						if(getch())
						main_menu();
					}
			    }
		    }
	    }  
	protected:
		void edit_menu()
		{
			char ch;
			while (1)
			{
				system("cls");
				cout <<"\n\n\t\t\t\t EDIT BOOKS\n";
				cout <<"\t\t\t\t~~~~~~~~~~~~\n\n" ;
				cout <<"\t\t\t1. Modify\n" ;
				cout <<"\t\t\t2. Delete\n" ;
				cout <<"\t\t\t0. Main Menu\n\n" ;
				cout <<"\t\t\tEnter Your Choice : " ;
				cin>>ch;
				switch(ch)
				{
					case '1':
						modify_books();
						break;
					case '2':
						delete_books();
						break;
					case '0':
						main_menu();
						break;
					default:
					{
						cout<<"\t\t\t\a\aWrong Entry!!\n";
						cout<<"\t\t\tPlease Enter Correct Option";
						if(getch())
						edit_menu();
					}
				}
			}
		}
		void issue_return()
		{
			char ch1;
			while (1)
			{
				system("cls");
				cout <<"\n\n\t\t\t\t ISSUE/RETURN BOOKS\n";
				cout <<"\t\t\t\t ~~~~~~~~~~~~~~~~~~\n\n" ;
				cout <<"\t\t\t1. Issue Books\n" ;
				cout <<"\t\t\t2. Return Books\n" ;
				cout <<"\t\t\t0. Main Menu\n\n" ;
				cout <<"\t\t\tEnter Your Choice : " ;
				cin>>ch1;
				switch(ch1)
				{
					case '1':
						issue_books();
						break;
					case '2':
						return_books();
						break;
					case '0':
						main_menu();
						break;
					default:
					{
						cout<<"\t\t\t\a\aWrong Entry!!\n";
						cout<<"\t\t\tPlease Enter Correct Option";
						if(getch())
						issue_return();
					}
				}
			}
		}
		
};

// Function To Add Books
void BookArchive::add_books()
{
	char choice;
	do
    {
    	system("cls");
		cin.ignore();
    	cout<<"\tADD BOOKS TO LIBRARY";
    	cout<<"\nNEW BOOK ENTRY...";
	    cout<<"\n\nEnter Book Code: ";
	    getline(cin,bookcode[bookcount]);
	    cout<<"Enter Name of Book: ";
	    gets(name[bookcount]);
	    cout<<"Enter Author's Name: ";
	    gets(author[bookcount]);
	    cout<<"Enter Price: ";
	    cin>>(price[bookcount]);
	    cout<<"Enter Quantity: ";
	    cin>>(quantity[bookcount]);
	    cout<<"\n\nBOOK CREATED!!";
	    issue[bookcount]=0;
	    bookcount++;
	    cout<<"\n\nEnter Another Book?(y/n): ";
	    cin>>choice;
    }
    while (choice=='Y' || choice=='y');
    cout<<"Press any key to go back to Main Menu...";
    getch();
}
// Function To Show Books List
void BookArchive::list_of_books()
{
	system("cls");
	cout<<"\tLIST OF ALL LIBRARY BOOKS";
	for(i=0 ; i<bookcount ; i++)
	{
		cout<<endl<<endl;
		cout<<"Book Code: ";
		cout<<(bookcode[i])<<endl;
		cout<<"Book Name: ";
		puts(name[i]);
		cout<<"Author Name: ";
		puts(author[i]);
		cout<<"Price: "<<price[i]<<endl;
		cout<<"Quantity: "<<quantity[i]<<endl<<endl;
	}
	if(i==0)
	cout<<"\nNo Book Record.\n\n";
	cout<<"Press any key to go back to Main Menu...";
    getch();
}

// Function To Modify Books
void BookArchive::modify_books()
{
	char choice;
	string bk_nmbr;
	do
	{
		system("cls");
		cin.ignore();
		cout<<"\tBOOK MODIFYING PANEL\n\n";
		cout<<"Enter Book Code to Modify: ";
		getline(cin,bk_nmbr);
		cout<<"Searching....\n";
		Sleep(5000);
		for(i=0 ; i<bookcount ; i++)
		if(bk_nmbr==bookcode[i])
		{
			cout<<"\n\tBOOK RECORD FOUND!!\n\n";
			cout<<"Add Modifying Details\n";
			cout<<"Enter Book Code: ";
		    getline(cin,bookcode[i]);
		    cout<<"Enter Name of Book: ";
		    gets(name[i]);
		    cout<<"Enter Author's Name: ";
		    gets(author[i]);
		    cout<<"Enter Price: ";
		    cin>>(price[i]);
		    cout<<"Enter Quantity: ";
		    cin>>(quantity[i]);
		    cout<<"\n\nBook with Code '"<<bookcode[i]<<"' has been Modified..";
			break;
		}
		if(i==bookcount)
		cout<<"\n\nNo Such Book Record is Found.";
		cout<<"\nModify Another Book?(y/n): ";
		cin>>choice;
	}
	while (choice=='y' || choice=='Y');
	cout<<"Press any key to go back to Edit Menu...";
    getch();
}



// Function To Delete Books

void BookArchive::delete_books()
{
	char choice;
	string bk_nmbr;
	do
	{
		
		system("cls");
		cin.ignore();
		cout<<"\tBOOK DELETING PANEL\n\n";
		cout<<"Enter Book Code to be Deleted:";
		getline(cin,bk_nmbr);
		cout<<"Searching....\n";
		Sleep(5000);
		for(i=0 ; i<bookcount ; i++)
		if(bk_nmbr==bookcode[i])
			{
				cout<<"\n\tBOOK RECORD FOUND!!\n\n";
				for(j=i ; j<bookcount ; j++)
					{
						issue[j]=issue[j+1];
						bookcode[j]=bookcode[j+1];
						strcpy(name[j],name[j+1]);
						strcpy(author[j],author[j+1]);
						price[j]=price[j+1];
						quantity[j]=quantity[j+1];
					}
				bookcount-=1;
				cout<<"\n\n\nBook Record Deleted Successfully...\n";
				break;
			}
	if(i==bookcount)
	cout<<"\n\nNo Such Book Record is Found.";
	cout<<"\nDelete Another Book?(y/n): ";
	cin>>choice;
	}
	while(choice=='y' || choice=='Y');
	cout<<"Press any key to go back to Edit Menu...";
    getch();
}


// Function To Search Books
void BookArchive::search_books()
{
	char choice;
	string bk_nmbr;
	do
	{
		system("cls");
		cin.ignore();
		cout<<"\tBOOK SEARCHING PANEL\n\n";
		cout<<"Enter Book Code to Search: ";
		getline(cin,bk_nmbr);
		cout<<"Searching....\n";
		Sleep(5000);
		for(i=0 ; i<bookcount ; i++)
		if(bk_nmbr==bookcode[i])
		{
			cout<<"\n\tBOOK RECORD FOUND!!\n\n";
			cout<<"Complete Book Details\n";
			cout<<"Book Code: ";
			cout<<(bookcode[i])<<endl;
			cout<<"Book Name: ";
			puts(name[i]);
			cout<<"Author Name: ";
			puts(author[i]);
			cout<<"Price: "<<price[i]<<endl;
			cout<<"Quantity: "<<quantity[i]<<endl;
			break;
		}
		if(i==bookcount)
		cout<<"\n\nNo Such Book Record is Found.";
		cout<<"\nSearch Another Book?(y/n): ";
		cin>>choice;
	}
	while (choice=='y' || choice=='Y');
	cout<<"Press any key to go back to Main Menu...";
    getch();
}
// Function To Issue Books
void BookArchive::issue_books()
{
	char choice, choice1;
	string bk_nmbr;
	do
	{
		system("cls");
		cin.ignore();
		cout<<"\tBOOK ISSUING PANEL\n\n";
		cout<<"Enter Book Code to Issue: ";
		getline(cin,bk_nmbr);
		cout<<"Searching....\n";
		Sleep(5000);
		for(i=0 ; i<bookcount ; i++)
		if(bk_nmbr==bookcode[i])
		{
			cout<<"\n\tBOOK RECORD FOUND!!\n\n";
			cout<<"Complete Book Details\n";
			cout<<"Book Code: ";
			cout<<(bookcode[i])<<endl;
			cout<<"Book Name: ";
			puts(name[i]);
			cout<<"Author Name: ";
			puts(author[i]);
			cout<<"Price: "<<price[i]<<endl;
			cout<<"Quantity: "<<quantity[i]<<endl<<endl;
			if (quantity[i]>0)
			{
				cout<<"Do You Want to Issue This Book?(y/n): ";
				cin>>choice1;
				if (choice=='y' || choice=='Y')
				issue[i]==1;
				quantity[i]--;
				cout<<"\nBook with Code '"<<bookcode[i]<<"' is Issued\n";
			}
			else
			{
				cout<<"This Book is Out of Stock\n";
			}
			break;
		}
		if(i==bookcount)
		cout<<"\n\nNo Such Book Record is Found.\n";
		cout<<"\nSearch Another Book?(y/n): ";
		cin>>choice;
	}
	while (choice=='y' || choice=='Y');
	cout<<"Press any key to go back to Issue/Return Menu...";
    getch();
}
// Function To Return Books
void BookArchive::return_books()
{
	char choice, choice1;
	string bk_nmbr;
	do
	{
		system("cls");
		cin.ignore();
		cout<<"\tBOOK RETURNING PANEL\n\n";
		cout<<"Enter Book Code to Return: ";
		getline(cin,bk_nmbr);
		cout<<"Searching....\n";
		Sleep(5000);
		for(i=0 ; i<bookcount ; i++)
		if(bk_nmbr==bookcode[i])
		{
			cout<<"\n\tBOOK RECORD FOUND!!\n\n";
			cout<<"Complete Book Details\n";
			cout<<"Book Code: ";
			cout<<(bookcode[i])<<endl;
			cout<<"Book Name: ";
			puts(name[i]);
			cout<<"Author Name: ";
			puts(author[i]);
			cout<<"Price: "<<price[i]<<endl;
			cout<<"Quantity: "<<quantity[i]<<endl<<endl;
			cout<<"Do You Want to Return This Book?(y/n): ";
			cin>>choice1;
			if (choice=='y' || choice=='Y')  
			issue[i]==0;
			quantity[i]++;
			cout<<"\nBook with Code '"<<bookcode[i]<<"' has been Returned\n";
			break;
		}
		if(i==bookcount)
		cout<<"\n\nNo Such Book Record is Found.\n";
		cout<<"\nSearch Another Book?(y/n): ";
		cin>>choice;
	}
	while (choice=='y' || choice=='Y');
	cout<<"Press any key to go back to Issue/Return Menu...";
    getch();
}
int main()
{
	Library L;
	L.introduction();
	L.main_menu();
}   