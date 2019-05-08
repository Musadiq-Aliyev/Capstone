package ELearning;

public class Admin 
{
	private static int user_id;
	private static String name;
	private static String userName;
	private static int age;

	//methods for the admin to perform the certain tasks
	private static void setName(String username)
	{
		System.out.println("The user name has been set to : " + username );
		userName = username;
	}
	
	private static void deleteData(int id)
	{
		System.out.println("The required data against id:  " + id + "  has been deleted");
		
	}
	private static void getName()
	{
		System.out.println("The name of the user is " + userName);
	}
	
	public static void main(String args[])
	{
		setName("Elvin");
		deleteData(5536125);
		getName();
	}

}
