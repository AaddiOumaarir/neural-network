#include<iostream>
#include<string>
#include<vector>
using namespace std;

class Employee{
    public:
    string name;
    int id;

    void set_salary(int potentialSalary){


        salary = potentialSalary;

    }
    void get_salary(){
        cout<<"The salary of "<< name << " is "<< salary<<" ";
    }
    
    private:
    float salary;

};

int main(){

    Employee employee1;
    employee1.name = "Addi";
    employee1.id = 15;
    employee1.set_salary(200000);

    cout << employee1.name << " is the "<< employee1.id << " th employee of this company"<<endl;
    employee1.get_salary();

    system("pause>0");
}