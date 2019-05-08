package ELearning;

public class Payment 
{
	private static double amount;
	private static int course_ID;
	private static String Name;
	private static int Student_ID;
	
	//methods
	
	private static void makePayment(int studentId, double amount,int courseID)
	{
		System.out.println("The Student : " +studentId + " have Paid the amount of the "+amount+ " of the course : " + courseID);
	}
	public static void main(String[] args) 
	{
		makePayment(5536125,5500.215, 36144);
	}

}
