package ELearing;

public class Video 
{
	private static byte studentId;
	private static byte ID;
	private static String Name;
	private static String UploadedBy;
	
	//methods
	
	//uploading video information 
	private static void UploadVideo(String VideoTitle ,String Name)
	{
		System.out.println("This Video : " + VideoTitle + "Is Uploaded By "+ Name);
	}
	
	//deleted indformation 
	
	private static void deletedVideo(String Deleted_Video_Name)
	{
		System.out.println("This Video of title " + Deleted_Video_Name +"Is deleted Successfully");
	}
	
	//edit the video 
	
	private static void editVideo_Info(String videoName)
	{
		System.out.println("The Video Name : "+ videoName + " Edited");
	}
	
	//main  class 
	
	public static void main(String args[])
	{
		UploadVideo("Hello World", Name);
		deletedVideo("Musadiq Aliyev");
		editVideo_Info("How to design");
	}
}
