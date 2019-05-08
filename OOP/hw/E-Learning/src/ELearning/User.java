package ELearning;

public class User 
{
	private static String profile_info;
	private static int ID;
	private static String userName;
	
	//constructor 
	
	User()
	{
		this.profile_info = "Student";
		this.ID= 5536125;
		this.userName = "Habil";
		
	}
	
	//main method 
	
	public static void main(String args[])
	{
		User user = new User();
		System.out.println("Users Info is : " + user.profile_info + " and user Name is :  " 
		+ user.userName + " and ID is : " + user.ID);
	}
}
