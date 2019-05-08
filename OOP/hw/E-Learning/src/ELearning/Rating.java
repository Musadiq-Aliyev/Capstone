package ELearning;

import java.util.Scanner;

public class Rating
{
	private static int id;
	private static String CourseName;
	private static String Rateby;
	private static String FeedBack;
	Scanner scan = new Scanner(System.in);
	
	 private static void give_Feedback(String feedback , String feedback_by )
	 {
		 System.out.println("Kindly provide the feedback");
		 FeedBack = feedback;
		 Rateby = feedback_by;
		 
		 System.out.println("This Feedback is : " + FeedBack + " :  given by : " + Rateby );
	 }
	public static void main(String[] args) 
	{
		give_Feedback("Nice Working with you", "Elvin");
	}

}
