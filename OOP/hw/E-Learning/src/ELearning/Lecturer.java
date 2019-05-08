package ELearning;

public class Lecturer 
{
	//attributes
	private String experience ;
	private int ID;
	private static String Name;
	private byte number_of_Courses;
	
	//constructor
	public Lecturer() 
	{
		Name = "User";
	}
	
	//methods 
	
	//method one of the RecievePayment
	private static void RecievePayment(int cash, String NameOf_Reciever)
	{
		if(NameOf_Reciever == Name)
		{
			System.out.println("cash Paid to  : " + cash);
		}
	}
	
	//method 2 of the setting the user name 
	private static void setName(String name)
	{
		Lecturer.Name = name;
		System.out.println("Name has been Updated to : " + Lecturer.Name );
		
	}
	
	//method 3 of the View Student
	private static void ViewStudent(String name)
	{
		System.out.println(name);
	}
	
	public static void main (String args[])
	{
		RecievePayment(3000,"User");
		setName("Musadiq");
		ViewStudent("Students");
		
	}
}
