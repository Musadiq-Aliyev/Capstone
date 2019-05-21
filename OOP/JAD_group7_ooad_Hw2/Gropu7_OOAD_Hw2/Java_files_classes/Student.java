package ELearing;

public class Student
{
	//attributes
	private int ID;
	private String Name ;
	private int age ;
	private  String gender;
	private String Course_Learned;

	//methods
	
	//1st method to Pay For the Course
	
	private static void PayFor_Course(int cash)
	{
		System.out.println("Fee of the Course is paid :" + cash);
	}

	//2nd method is for the register the student
	
	private static void Register_Course(String name , int id)
	{
		System.out.println(name + " has been registered with the id : ("+ id +") Successfully ");
	}
	public static void main(String args[])
	{
		PayFor_Course(1000);
		Register_Course("Elvin", 5536125);
	}
}
