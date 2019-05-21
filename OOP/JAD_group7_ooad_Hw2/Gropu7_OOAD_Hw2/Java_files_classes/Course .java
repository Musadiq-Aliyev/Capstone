package ELearing;

public class Course 
{
	private static String category;
	private static int ID;
	private static String Name;
	private static String rating;
	private static String UploadedBy;
	
	
	//methods definition
	
	private static void deleteCourse(int Id)
	{		
		System.out.println(Id +"  ID Course Deleted");
	} 
	
	private static void rankCourse(int id, String name,String rating)
	{
		System.out.println("The course :  " + id + " Name : " + name + "is been rated by rating : " + rating  );
	}
	
	private static void registerForCourse(String coursename,String name, int id )
	{
		System.out.println("ID : " + id + " and Name :"+ name + "  has register for the course  " + coursename + "  successfully");
	}
	private static void uploadCourse(String name, int id)
	{
		System.out.println("New course : " + name + "is uploaded with the id : " + id+ " successfully");
	}
	public static void main(String[] args) 
	{
		deleteCourse(5536125);
		rankCourse(5536125, "OOP","10");
		registerForCourse("OOP", "Elvin", 5536125);
		uploadCourse("OOP " , 5536125);
	}

}
