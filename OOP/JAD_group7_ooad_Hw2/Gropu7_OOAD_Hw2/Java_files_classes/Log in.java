package ELearing;

public class Log_In 
{
	//attributes
	private int id;
	private static String userName;
	private static String password;
	
	//constructor for the default values setting 
	Log_In()
	{
		this.id = 5536125;
		this.userName = "Musadiq";
		this.password = "i love animals";
		
	}
	//methods
	
	//1st method authetication of theuserName and the passward
	 private static void user_Authentication(String userName, String Passward)
	 {
		 if(userName == Log_In.userName || Passward == Log_In.password)
		 {
			 System.out.println("Authenticated User ");
		 }
		 else
		 {
			 System.out.println("Sorry wrong  Passward and userName");
		 }
	 }
	 
	 public static void main(String args[])
	 {
		 Log_In log = new Log_In();
		 user_Authentication("Musadiq","i love animals" );
	 }
}
