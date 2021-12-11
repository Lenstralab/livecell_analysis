/*   Macro for Creating a Tracking text file

Version:  See below
Description: This macro has the option to:
   * Track cells by (clicking [manual] or follow [semi-automatic])
   * Create crop image from single cell/area
   * Create mask image of the crop image (segmentation image)
   * 
*/
//  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  
// Macro credentials
	title		=	"Tracking";
	version		=	"v3";
	MACRO 		= 	title+version;
	date		=	"30 Mar 2020";
	Contact		= 	"l.joosen@nki.nl"
	Internet	= 	"https://spark.adobe.com/page/RQ742xJIRlNpF/";
//  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  ///  //  

macro title {

//===> 0    <========= Start =======================================================
//===> 0.1    <========= Start - Variables =======================================================
//========================================================================================
// Standards/Settings
CropImage = "Yes";
	// miscelanious
NrChannels 	= 2;								// amount of channels, set as default in dialog screen
delimiter 	= "/";								// folder devider (e.g.: /Users/Desktop/ = /). for Mac: /, for PC \.
delimiter2 	= "_";								// name devider (e.g.: 20190715_Axio_TCNER, splits into 20190715.
Bkgr 		= 100;								// background subtraction (rolling ball method) on image for drawing segment region
SumName 	= "_sum_cell_mask";					// name to use for extension for segmentation image
SelectRegion = 1;								// if each cell contains just one single spot to analyse
DefSelChannel = 2;								// Default channel for selecting cells
DefSpeedFPS = 3;								// Default speed for tracking (when follow is selected)

//===> 0.2    <========= Start - Dialog =======================================================
	nIm = nImages;
	Radius = 11;	
	StartTime = (getTime());
	if (nIm == 1) {
		OpenIm = "Current File";
		getDimensions(width, height, Channels, slices, frames);
		if (Channels > 3) {
			NrChannels = 2;
			OpenIm = "Open Stack";
		}
		Radius = round(20000 / width);	// preselected radius (for method = click) = 2% 	
		filedir = getDirectory("image");
		FileName = getTitle();
	} else {
		OpenIm = "Open Stack";
	}
	if (NrChannels == 1) { 	DefCol1 = "Grays";	DefCol2 = "Not"; 	DefCol3 = "Not"; 	}
	if (NrChannels == 2) { 	DefCol1 = "Red";	DefCol2 = "Green"; 	DefCol3 = "Not"; 	}
	if (NrChannels == 3) {	DefCol1 = "Red";	DefCol2 = "Green"; 	DefCol3 = "Blue"; 	}
	
	// Mic = "Andor SpinningDisk";
	
	// Dialogs
	Dialog.create(""+title+", "+version+" ::-:: Start ");
		Dialog.addMessage("<<  1  >>       Open Images :                                                          ") ;
		items = newArray("Current File", "Open Sequence", "Open Stack");
	Dialog.addRadioButtonGroup("        1  A         Work on Image : ", items, 1, 3, OpenIm);
	if (OpenIm == "Current File") {
		Dialog.addMessage("       ->  Image Name :            " + FileName) ;			
		Dialog.addMessage("       ->  Image Directory :       " + filedir) ;			
		Dialog.addMessage("       ->  Current Image :          Size = " + width + "x" + height + ",   Channel = " + Channels + ",   Slices = " + slices + ",   Frames = " + frames ) ;			
	}
	//	items = newArray("Zeiss AxioObserver", "Andor SpinningDisk");
	// 	Dialog.addRadioButtonGroup("Microscope Used : ", items, 1, 2, Mic);
	Dialog.addNumber("           1  B         Amount of Channels  : ", NrChannels, 0, 10, "  ");
	Dialog.addChoice("       Color Channel 1 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), DefCol1);
		Dialog.addToSameRow();
	Dialog.addChoice("Channel 2 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), DefCol2);
		Dialog.addToSameRow();
	Dialog.addChoice("Channel 3 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), DefCol3);
		Dialog.addMessage("") ;
		Dialog.addMessage("<<  2  >>       Track, Crop & Segment :                                                          ") ;
		items = newArray("Yes", "No");
	Dialog.addRadioButtonGroup("          2  A         Track Single Cells : ", items, 1, 2, "Yes");
		items = newArray("Yes", "No");
	Dialog.addRadioButtonGroup("          2  B         Crop Images ( only if tracking is performed ) : ", items, 1, 2, "No");
		items = newArray("Yes", "No");

	if (OpenIm == "Current File") {
		CropWidth = round(width / 5);			CropWith2 = round(CropWidth / 2);
		w2 = width / 2 - CropWith2;		h2 = height / 2 - CropWith2;
		makeRectangle(w2, h2, CropWidth, CropWidth);
		getVoxelSize(width, height, depth, pixunit);
		Dialog.addNumber("  Crop size : ", CropWidth, 0, 10, " ( Only if cropped. see image [1/5] )");
	}
	Dialog.addRadioButtonGroup("          2  C         Segment/Mask Image ( only if tracking & crop is performed ) : ", items, 1, 2, "No");
	Dialog.addChoice("       Selection Tool :",newArray("rectangle", "oval", "polygon", "freehand"), "freehand");
		Dialog.addMessage("") ;
	crop = 1;
	Dialog.addMessage("        2  D         Tracking Settings");
	if (OpenIm == "Current File") {
		getDimensions(width, height, channels, slices, frames);
		Dialog.addMessage("                                    ⇣ 0 : no ROI will be drawn ");
		Dialog.addNumber("  Draw ROIs  : ", crop, 0, 10, "   ( Nr cells )");
		Dialog.addToSameRow();
			totalTimeFPS = frames / DefSpeedFPS;
			if (totalTimeFPS < 60) { FPSconst = " sec";		totalTimeFPS = d2s(totalTimeFPS, 1);	}
			if (totalTimeFPS >= 60) { FPSconst = " min";	totalTimeFPS = d2s(totalTimeFPS / 60, 1);	}
		Dialog.addNumber("     FPS [if: Follow] : ", DefSpeedFPS, 1, 10,"   ( = " + totalTimeFPS + FPSconst + " (if = "+DefSpeedFPS +") )");
		items = newArray("Click", "Follow");
		Dialog.addChoice("       Method :",newArray("Click", "Follow"), "Follow");
			Dialog.addToSameRow();
		Dialog.addNumber("     Radius [if: Click] : ", Radius, 0, 10,"   ( "+ pixunit+" = 2% of image )");
		Dialog.addMessage("") ;
			if (NrChannels == 1) {	items = newArray("Channel 1");	NrRadioBut = 1;	ChRadioBut = "Channel 1";	}
			if (NrChannels == 2) {	items = newArray("Channel 1", "Channel 2", "All/Overlay");	NrRadioBut = 3;	ChRadioBut = "All/Overlay";	}
			if (NrChannels == 3) {	items = newArray("Channel 1", "Channel 2", "Channel 3", "Overlay/All");	NrRadioBut = 4;	ChRadioBut = "Overlay/All";	}
		Dialog.addRadioButtonGroup("            2  E         Draw from Channel : ", items, 1, NrRadioBut, ChRadioBut);
	}
		Dialog.addMessage("<<  3  >>       Save :                                                          ") ;
				rows = 2;  columns = 4; n = rows*columns; labels = newArray(n); defaults = newArray(n);
				labels[0] ="";  labels[1] = "Don't Check";			labels[2] = "After Each Tracking";		labels[3] = "At the End";		
				labels[4] = "";	labels[5] ="Save Track Image";	labels[6] = "";				labels[7] = ""; 
				defaults[1] = false;  
				defaults[2] = true;  
				defaults[3] = true; 
				defaults[5] = true; 
				Dialog.addCheckboxGroup(rows,columns,labels,defaults);
		Dialog.addMessage("                                     If: Save track image is selected = Check: At the End is selected too");
//		Dialog.addMessage("<<  3  >>       Save :                                                          ") ;
//		items = newArray("No", "After Each Tracking", "At the End");
//	Dialog.addRadioButtonGroup("            3  A         View Tracked Spots (after Tracking ) : ", items, 1, 2, "After Each Tracking");

//		Dialog.addMessage("<<  3  >>       Save Data :                                                          ") ;
//	Dialog.addChoice("       Input Folder :",newArray(InputDir1, InputDir2, InputDir3), InputDir3);
//		Dialog.addToSameRow();
//	Dialog.addChoice("       Output :",newArray("Folder : "+OutputDir1, "Folder : "+OutputDir2, "Folder : "+OutputDir3, "Same Folder", "Subfolder -> Data", "Subfolder -> Analyzed", "Determine"), "Folder : "+OutputDir2);
//		Dialog.addMessage("Save Folder : ") ;
//		rows = 1;  columns = 3; n = rows*columns; labels = newArray(n); defaults = newArray(n);
//		labels[0]  = "";  labels[1]  = "Log File";  labels[2] = "ROI zip";	
		//	labels[3] = "Sum Image";	labels[4]  = "";  labels[5] = "";  labels[6] = "ROI";	labels[7] = "Log File";
//		defaults[1] = true;	
		//	defaults[1] = false;	defaults[3] = true;	defaults[5] = false;	defaults[6] = true;	defaults[7] = true;
//		Dialog.addCheckboxGroup(rows,columns,labels,defaults);
		if (OpenIm != "Current File") {	Dialog.addMessage("") ; }
	Dialog.show();
	// 	0.2.2		Dialogs Settings
	// << 1 >> Open Images
	WorkOn			= Dialog.getRadioButton();
	// Microscope	 	= Dialog.getRadioButton();
	Channels 		= Dialog.getNumber();
	ColorChannel1	= Dialog.getChoice();
	ColorChannel2	= Dialog.getChoice();
	ColorChannel3	= Dialog.getChoice();
	Track 			= Dialog.getRadioButton();
	Crop 			= Dialog.getRadioButton();
	if (OpenIm == "Current File") {
		CropWidth		= Dialog.getNumber();	CropHeight = CropWidth;
	}
	Sum				= Dialog.getRadioButton();
	SelectTool		= Dialog.getChoice();
	if (OpenIm == "Current File") {		
		SelectRegion	= Dialog.getNumber();		// amount of regions to track
		FPS				= Dialog.getNumber();		// frames per second
		SelectMethod	= Dialog.getChoice();		// method for tracking
		Radius 			= Dialog.getNumber();		Radius = Radius / 2;	// size of circle to use for tracking in pixels
		SelectChannel	= Dialog.getRadioButton();	// perform selection of tracking on which channel. creates Channel...
	}
//	InputDir		= Dialog.getChoice();
//	SaveData		= Dialog.getChoice();
//	SaveLog			= Dialog.getCheckbox();		if (SaveLog == 1) { SaveLog = "Yes"; }	if (SaveLog == 0) { SaveLog = "No"; }
//	SaveROI			= Dialog.getCheckbox();		if (SaveROI == 1) { SaveROI = "Yes"; }	if (SaveROI == 0) { SaveROI = "No"; } 
	SaveLog = "Yes";	// is replaced to be automatic
	SaveROI = "Yes";	// is replaced to be automatic
//	TrackView = Dialog.getRadioButton();
	CheckNot = Dialog.getCheckbox();
	CheckEach = Dialog.getCheckbox();	 if (CheckEach == true) { CheckNot = false;}
	CheckEnd = Dialog.getCheckbox();	 if (CheckEnd == true) { CheckNot = false;}
	SaveImage = Dialog.getCheckbox();

	CDialog = "No";
	ROIcol = "Red";
	if (Channels == 1) { 	ColorCh1 = ColorChannel1;
							if (ColorChannel1 == "Not") {	CDialog = "Yes";	ColorChannel1 = "Grays";	}
							ROIcol = "Green";
	}
	if (Channels == 2) { 	ColorCh1 = ColorChannel1;	ColorCh2 = ColorChannel2;
							if (ColorChannel1 == "Not") {	CDialog = "Yes";	ColorChannel1 = "Red";		}
							if (ColorChannel2 == "Not") {	CDialog = "Yes";	ColorChannel2 = "Green";	}
							if (OpenIm == "Current File") {	if (SelectChannel == "Channel 1") { ROIcol = ColorCh1; }		if (SelectChannel == "Channel 2") { ROIcol = ColorCh2; }		if (SelectChannel == "Overlay/All") { ROIcol = ColorChannel1; }	}
	}
	if (Channels == 3) { 	ColorCh1 = ColorChannel1;	ColorCh2 = ColorChannel2;	ColorCh3 = ColorChannel3;
							if (ColorChannel1 == "Not") {	CDialog = "Yes";	ColorChannel1 = "Red";		}
							if (ColorChannel2 == "Not") {	CDialog = "Yes";	ColorChannel2 = "Green";	}
							if (ColorChannel3 == "Not") {	CDialog = "Yes";	ColorChannel3 = "Blue";		}
							if (SelectChannel == "Channel 1") { ROIcol = ColorCh1; }		if (SelectChannel == "Channel 2") { ROIcol = ColorCh2; }		if (SelectChannel == "Channel 3") { ROIcol = ColorCh3; }		if (SelectChannel == "Overlay/All") { ROIcol = ColorChannel1; }
	}


//===> 0.3    <========= Start - Dialog 1 image  =======================================================
// if image was not yet open
	if (CDialog == "Yes") {
		Dialog.create(""+title+", "+version+" ::-:: Define colors ");
		Dialog.addMessage("The amount of channels >> " + Channels + " <<, is not consistent with channel colors");
		Dialog.addMessage("Define channel colors correctly");
		if (Channels == 1) { 	Dialog.addMessage("Current chosen colors : ch 1 = " + ColorCh1);	}
		if (Channels == 2) { 	Dialog.addMessage("Current chosen colors : ch 1 = " + ColorCh1 + "   /   ch 2 = " + ColorCh2);	}
		if (Channels == 3) { 	Dialog.addMessage("Current chosen colors : ch 1 = " + ColorCh1 + "   /   ch 2 = " + ColorCh2 + "   /   ch 3 = " + ColorCh3);	}
		Dialog.addMessage("");	Dialog.addMessage("");
		Dialog.addNumber("  Amount of Channels  : ", Channels, 0, 10, "  ");
		Dialog.addChoice("       Color Channel 1 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), ColorChannel1);
			Dialog.addToSameRow();
		Dialog.addChoice("Channel 2 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), ColorChannel2);
			Dialog.addToSameRow();
		Dialog.addChoice("Channel 3 :",newArray("Red", "Green", "Blue", "Cyan", "Magenta", "Yellow", "Grays", "Not"), ColorChannel3);
		Dialog.show();
		Channels 		= Dialog.getNumber();
		ColorChannel1	= Dialog.getChoice();
		ColorChannel2	= Dialog.getChoice();
		ColorChannel3	= Dialog.getChoice();
		
		if (Channels == 1) { 	ColorCh1 = ColorChannel1;
								if (ColorChannel1 == "Not") {	CDialog = "Yes";	ColorChannel1 = "Grays";	}
								ROIcol = "Green";
		}
		if (Channels == 2) { 	ColorCh1 = ColorChannel1;	ColorCh2 = ColorChannel2;
								if (ColorChannel1 == "Not") {	CDialog = "Yes";	ColorChannel1 = "Red";		}
								if (ColorChannel2 == "Not") {	CDialog = "Yes";	ColorChannel2 = "Green";	}
								if (SelectChannel == "Channel 1") { ROIcol = ColorCh1; }		if (SelectChannel == "Channel 2") { ROIcol = ColorCh2; }		if (SelectChannel == "Overlay/All") { ROIcol = ColorChannel1; }
		}
	}

//===> 0.4    <========= Start - Open File =======================================================
//===> 0.4.1    <========= Start - Open File - Current File =======================================================
	if (WorkOn == "Current file") {
		if (nImages != 1) {
			Dialog.create(""+title+", "+version+" ::-:: Image Composition ");
				Dialog.addMessage("Amount of images >> " +  nImages + ", can't be used for analysis, open a new image");
				Dialog.addMessage("<<  1  >>       Open Images :                                                          ") ;
				items = newArray("Open Sequence", "Open Stack");
			Dialog.addRadioButtonGroup("        1  A         Work on Image : ", items, 1, 2, "Open Stack");
			Dialog.show();
			WorkOn			= Dialog.getRadioButton();
		}
		if (nImages >= 1) {
			getDimensions(width, height, channels, slices, frames);
			// directory settings
			filedir = getInfo("image.directory")+getInfo("image.filename");
			filedir = substring(filedir,0,lengthOf(filedir)-4)+"";
			FileName = getTitle();
			if (endsWith(filedir, FileName)) {	// delete filename from directory in path
				dotIndex = indexOf(filedir, FileName); 
				filedir = substring(filedir, 0, dotIndex); 
			}

			FileName = getTitle();
			separatorIMS = ".ims";
			dotIndex = indexOf(FileName, separatorIMS);
			if (dotIndex > 0 )  {	// delete .ims if present in filename
				FileName = substring(FileName, 0, dotIndex);
			}
			totalslice = channels * slices * frames;
			if (totalslice <= 1) { 
				Dialog.create(""+title+", "+version+" ::-:: Open Images ");
					Dialog.addMessage("<<  1  >>       Open Images :                                                          ") ;
					items = newArray("Open hyperstack", "Open sequence", "Current file");
				Dialog.addRadioButtonGroup("Work on : ", items, 1, 3, "Current file");
				Dialog.show();
				WorkOn=Dialog.getRadioButton();			
			}
		}
	}

//===> 0.4.2    <========= Start - Open File - Not Current File =======================================================
	if (WorkOn != "Current file") {
		//Delete all open images and clear logfile
		run("Close All");
		roiManager("Reset");
		run("Clear Results"); 
	}
	if (isOpen("Results")) {
		selectWindow("Results"); 
		run("Close");
	}
	if (SaveLog == "Yes") {
		print ("+~=-=~+");
		selectWindow("Log");
		print ("\\Clear");
	}
	roi = roiManager("count");
	if (roi != 0) {
		roiManager("deselect");
		roiManager("delete");
	}

//===> 0.5    <========= Start - Print Date =======================================================
	print ("~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~~=>+<=~"); 
	//	1.3.1  Date & time:
	print ("Macro : ", title + ": " + version); 
	selectWindow("Log"); 
	//	1.3.1  Date & time:
	MonthNames = newArray("Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec");
	DayNames = newArray("Sun", "Mon","Tue","Wed","Thu","Fri","Sat");
	getDateAndTime(year, month, dayOfWeek, dayOfMonth, hour, minute, second, msec);
		if (DayNames[dayOfWeek] == "Sun") { DAY = "Sunday"; }	if (DayNames[dayOfWeek] == "Mon") { DAY = "Monday"; }	if (DayNames[dayOfWeek] == "Tue") { DAY = "Tuesday"; }	if (DayNames[dayOfWeek] == "Wed") { DAY = "Wednesday"; }
		if (DayNames[dayOfWeek] == "Thu") { DAY = "Thursday"; }	if (DayNames[dayOfWeek] == "Fri") { DAY = "Friday"; }	if (DayNames[dayOfWeek] == "Sat") { DAY = "Saturday"; }
		DateTimeString ="Date: "+DayNames[dayOfWeek]+" ";
		if (dayOfMonth<10) {DateTimeString = DateTimeString+"0";}
		DateTimeString = DateTimeString+dayOfMonth+"-"+MonthNames[month]+"-"+year+"     Time: ";
		if (hour<10) {DateTimeString = DateTimeString+"0";}
		DateTimeString = DateTimeString+hour+":";
		if (minute<10) {DateTimeString = DateTimeString+"0";}
		DateTimeString = DateTimeString+minute+":";
		if (second<10) {DateTimeString = DateTimeString+"0";}
		DateTimeString = DateTimeString+second;	
		//	print(DateTimeString);
		TimeStart = hour * 60;		TimeStart = TimeStart + minute * 60;	TimeStart = TimeStart + second;			
	// Date scientific
		DateString ="";
		if (month<10) {DateString = DateString+"0";}		Month = month + 1;						DateString = DateString+Month;
		if (dayOfMonth<10) {DateString = DateString+"0";}	DateString = DateString+dayOfMonth;
		Year = year - 2000;									YearString = "20" + Year;				DateString = YearString + DateString;
		//	print(DateString);	
	// Time scientific
		TimeString ="";
		if (hour<10) {TimeString = TimeString+"0";}			TimeString = TimeString + hour + "h";
		if (minute<10) {TimeString = TimeString+"0";}		TimeString = TimeString + minute + "m";
		if (second<10) {TimeString = TimeString+"0";}		TimeString = TimeString + second + "s";
	print ("Dimensions desktop screen : ", screenWidth, " * ", screenHeight);
	print ("Contact : ",Contact ,",   for more information  (check link in next line)");
	print(Internet);
	print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");

//===> 1      <========= Open =======================================================
	print ("1 # Image information :");
	print ("       * Image used :              " + WorkOn); 	
	// print ("       * Image used :              " + WorkOn + "   (" + Microscope + ")"); 	

//===> 1.1      <========= Open - Open Images =======================================================
//===> 1.1.1      <========= Open - Open Images - Current File =======================================================
	if (WorkOn == "Current file" ) {
		FileName = getTitle();
		filedir = getDirectory("image");
		filein=filedir+FileName;
	} // end of 'open' current image
//===> 1.1.2      <========= Open - Open Images - Not Current File =======================================================
	if (WorkOn != "Current file" ) {
		if(WorkOn == "Open Sequence"){
			path = File.openDialog("Select path");
			open(path);		
			filedir = getDirectory("image");	
			FileName = getTitle();
			rename("A");
			// Run Image Sequence (import images)		
			if(Contain != "N/A"){
				run("Image Sequence...", "open=&filedir file=0000 sort");
			}
			if(Contain == "N/A"){
				run("Image Sequence...", "open=&filedir sort");
				print ("       * File Contains :       ", Contain); 
			}
			rename(FileName);
			close("A");
	} // end of opening image sequence
		if(WorkOn == "Open Stack"){
			path = File.openDialog("Select path");
			open(path);		
			filedir = getDirectory("image");		
			FileName = getTitle();
		}
		filein=filedir+FileName;
	} // end of opening hyperstack

	getDimensions(width, height, channels, slices, frames);
	TotalSlices = channels * slices * frames;
	FramesHyper = TotalSlices / Channels;

//===> 1.2      <========= Open - Create Hyperstack =======================================================
	// Create hyperstack, if amount of channels is not equal to channels given
	if (channels != Channels) {
		run("Select All");
		waitForUser("Check Orientation of Stack, to Create Hyperstack");
		Dialog.create(""+title+", "+version+" ::-:: Define Channel Orientation ");
		Dialog.addNumber("  Amount of Channels  : ", Channels, 0, 10, "  ");
		Dialog.addMessage("");
			items = newArray("XY-C-Z-T", "XY-C-T-Z", "XY-Z-C-T", "XY-Z-T-C", "XY-T-C-Z", "XY-T-Z-C");
		Dialog.addRadioButtonGroup("Stack Orientation : ", items, 3, 2, "XY-C-Z-T");	
			
		Dialog.show();
		Channels = Dialog.getNumber();
		StackOrient = Dialog.getRadioButton();
		if (StackOrient == "XY-C-Z-T") { StackOrient = "xyczt(default)"; 	}
		if (StackOrient == "XY-C-T-Z") { StackOrient = "xyctz"; 	}
		if (StackOrient == "XY-Z-C-T") { StackOrient = "xyzct"; 	}
		if (StackOrient == "XY-Z-T-C") { StackOrient = "xyztc"; 	}
		if (StackOrient == "XY-T-C-Z") { StackOrient = "xytcz"; 	}
		if (StackOrient == "XY-T-Z-C") { StackOrient = "xytzc"; 	}
		run("Stack to Hyperstack...", "order="+StackOrient+" channels="+Channels+" slices=1 frames="+FramesHyper+" display=Color");	
		makeRectangle(w2, h2, CropWidth, CropWidth);
	}
	
//===> 1.3      <========= Open - Define Filename =======================================================
	separatorTIF = "";
	FilenameEnd = endsWith(FileName, ".tif");
	if (FilenameEnd == 1) {	separatorTIF = ".tif"; 	} 
	else { FilenameEnd = endsWith(FileName, ".tiff");
	if (FilenameEnd == 1) {	separatorTIF = ".tiff";	}
	}

	dotIndex = indexOf(FileName, separatorTIF);
	if (dotIndex > 0 )  {	// delete .tif(f) if present in filename
		FileName = substring(FileName, 0, dotIndex); 
		rename(FileName);
	} // end of renaming file when containing .tif or .tiff

//===> 1.4      <========= Open - Folder Output Name =======================================================
	print ("       * File Name :                ", FileName);
	print ("       * File directory :             ", filedir); 

	if (WorkOn != "Current file") {
		getDimensions(width, height, Channels, slices, frames);
		getVoxelSize(widthVox, heightVox, depth, pixunit);

		waitForUser("Check Amount of ROIs (=cells/spots) to track");
		Stack.setPosition(1, 1, 1);
		
		Dialog.create(""+title+", "+version+" ::-:: Image construction");
		Radius = round(20000 / width);	// preselected radius (for method = click) = 2% 	
		Dialog.addMessage("       ->  Image Name :            " + FileName) ;			
		Dialog.addMessage("       ->  Image Directory :       " + filedir) ;			
		Dialog.addMessage("       ->  Current Image :          Size = " + width + "x" + height + ",   Channel = " + Channels + ",   Slices = " + slices + ",   Frames = " + frames ) ;			
		Dialog.addMessage("");
		if (Crop == "Yes") {
			Dialog.addToSameRow();
			CropWidth = round(width / 5);			CropWith2 = round(CropWidth / 2);
			w2 = width / 2 - CropWith2;		h2 = height / 2 - CropWith2;
			makeRectangle(w2, h2, CropWidth, CropWidth);
			getVoxelSize(width, height, depth, pixunit);
			Dialog.addNumber("  Crop size : ", CropWidth, 0, 10, " ( Only if cropped. see image [1/5] )");
		}
		getDimensions(width, height, channels, slices, frames);
		Dialog.addMessage("                                    ⇣ 0 : no ROI will be drawn ");
		Dialog.addNumber("  Draw ROIs  : ", crop, 0, 10, "   ( Nr cells )");
			totalTimeFPS = frames / DefSpeedFPS;
			if (totalTimeFPS < 60) { FPSconst = " sec";		totalTimeFPS = d2s(totalTimeFPS, 1);	}
			if (totalTimeFPS >= 60) { FPSconst = " min";	totalTimeFPS = d2s(totalTimeFPS / 60, 1);	}
			Dialog.addToSameRow();
		Dialog.addNumber("     FPS [if: Follow] : ", DefSpeedFPS, 1, 10,"   ( = " + totalTimeFPS + FPSconst + " (if = "+DefSpeedFPS +") )");
		Dialog.addChoice("       Method :",newArray("Click", "Follow"), "Follow");
			Dialog.addToSameRow();
		Dialog.addNumber("     Radius [if: Click] : ", Radius, 0, 10,"   ( "+ pixunit+" = 2% of image )");
		Dialog.addMessage("") ;
			if (channels == 1) {
				chan1 = ColorChannel1;
				items = newArray(ColorChannel1);	NrRadioBut = 1;	ChRadioBut = "Channel 1";	
			}
			if (channels == 2) {
				chan1 = ColorChannel1;	chan2 = ColorChannel2;
				items = newArray(ColorChannel1, ColorChannel2, "All/Overlay");	NrRadioBut = 3;	ChRadioBut = "All/Overlay";	
			}
			if (channels == 3) {
				chan1 = ColorChannel1;	chan2 = ColorChannel2;	chan3 = ColorChannel3;
				items = newArray(ColorChannel1, ColorChannel2, ColorChannel3, "Overlay/All");	NrRadioBut = 4;	ChRadioBut = "Overlay/All";
			}
		Dialog.addRadioButtonGroup("            2  E         Draw from Channel : ", items, 1, NrRadioBut, ChRadioBut);
		Dialog.addMessage("") ;
		Dialog.show();
		if (Crop == "Yes") {
			CropWidth		= Dialog.getNumber();	CropHeight = CropWidth;
		}
		SelectRegion	= Dialog.getNumber();		// amount of regions to track
		FPS				= Dialog.getNumber();		// frames per second
		SelectMethod	= Dialog.getChoice();		// method for tracking
		Radius 			= Dialog.getNumber();	Radius = Radius / 2;	// size of circle to use for tracking in pixels
		SelectChannel	= Dialog.getRadioButton();	// perform selection of tracking on which channel. creates Channel...

		ROIcol = "Red";
		if (Channels == 1) { 	ROIcol = "Green";	
				if (SelectChannel == ColorChannel1) {	SelectChannel = "Channel 1"; }		
		}
		if (Channels == 2) { 	if (SelectChannel == "Channel 1") { ROIcol = ColorCh1; }		if (SelectChannel == "Channel 2") { ROIcol = ColorCh2; }		if (SelectChannel == "Overlay/All") { ROIcol = ColorChannel1; }
				if (SelectChannel == ColorChannel1) {	SelectChannel = "Channel 1"; }		if (SelectChannel == ColorChannel2) {	SelectChannel = "Channel 2"; }
		}
		if (Channels == 3) { 	if (SelectChannel == "Channel 1") { ROIcol = ColorCh1; }		if (SelectChannel == "Channel 2") { ROIcol = ColorCh2; }		if (SelectChannel == "Channel 3") { ROIcol = ColorCh3; }		if (SelectChannel == "Overlay/All") { ROIcol = ColorChannel1; }
				if (SelectChannel == ColorChannel1) {	SelectChannel = "Channel 1"; }		if (SelectChannel == ColorChannel2) {	SelectChannel = "Channel 2"; }		if (SelectChannel == ColorChannel3) {	SelectChannel = "Channel 3"; }
		}
	}
	
	if ( ROIcol == "Red") {		ROIcolor = "cyan"; }	if ( ROIcol == "Magenta") {	ROIcolor = "Green"; }	if ( ROIcol == "Yellow") {	ROIcolor = "blue"; }
	if ( ROIcol == "Green") {	ROIcolor = "magenta"; }	if ( ROIcol == "cyan") {	ROIcolor = "red"; }		if ( ROIcol == "Blue") {	ROIcolor = "yellow"; }	
	if ( ROIcol == "Grays") {	ROIcolor = "red"; }

//===> 1.5      <========= Open - Info =======================================================
//===> 1.5.1      <========= Open - Info - Dimensions of Original Image =======================================================
	getDimensions(wStack, hStack, cStack, sStack, fStack);
	if (sStack == 1) {
		Slices = 1;
	}
	getDimensions(width, height, Channels, Slices, Frames);
	TotalSlices = Channels * Slices * Frames;

	Stack.getUnits(X, Y, Z, Time, Value);
	Stack.getUnits(X, Y, Z, Time, Value)	if (X == "pixel") {X = "pixels";}	if (X == "micron") {X = "microns";}
	Bit = bitDepth();
	getVoxelSize(pixwidth, pixheight, pixdepth, pixunit);
	print ("       * Image Dimension :      channel: ", Channels, ",  slice: ", Slices, ",  frames: ", Frames);
	print ("       * Image Size Info :         (w) ", width, " x (h) ", height, X);
	print ("       * Pixel Size Info :        (w) ", pixwidth, " x (h) ", pixheight, pixunit);
	print ("       * Bit Depth :             ", Bit, " bit ");
	print ("       * Colors channel 1 :     ", ColorChannel1, ",  Channel 2 : ", ColorChannel2, ",  Channel 3 : ", ColorChannel3);

	// set contrast for each channel
	getDimensions(width, height, Channels, slices, frames);
	for (c = 1; c < Channels +1; c++) {
		Stack.setPosition(c, 1, 1);
		run("Enhance Contrast", "saturated=0.35");
		getMinAndMax(min, max);
		print("          - Image contrast : channel = ", c, " : min = ", floor(min), ", max = ",  floor(max));
	}

//===> 2     <========= Align =======================================================
	print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
	print ("2 # Image alignment :");
	print ("       * Parts performed :                    Track : ", Track, ",  Crop : ", Crop, ",  Segment/Mask : ", Sum);
	print ("       * Number of cells to track :          " + SelectRegion);
	if (Track == "Yes") {
		print ("          - Tracking method :     ", SelectMethod, "  ( Channel : ", SelectChannel, ")");		
	}
	if (Crop == "Yes") {
		print ("          - Crop size :     ", CropWidth, X);
	}
	if (Sum == "Yes") {
		print ("          - Segmentation tool :     ", SelectTool);
	}
//===> 2.1     <========= Align - Save Max Image =======================================================
	selectWindow(FileName);
	if (endsWith(filedir, "/")) {	filedir = filedir;	} 
	else {	filedir = filedir + "/";	}

	rename(FileName);
	roiManager("Show All");

//===> 2.2     <========= Align - Set Location of Image =======================================================
	run("Scale to Fit");
	ratio = width / height;					ImageHeight = screenHeight * 0.9;		ImageWidth = ImageHeight * ratio;	
	setLocation(15, 0, ImageWidth, screenHeight * 0.95);
	
	if (SelectChannel == "All/Overlay") {
		Stack.setDisplayMode("composite");
		SelectChan = Channels;
	}
	if (SelectChannel != "All/Overlay") {
		Stack.setDisplayMode("color");
		if (SelectChannel == "Channel 1") {	SelectChan = 1; } else { 
		if (SelectChannel == "Channel 2") {	SelectChan = 2; } else { 
		if (SelectChannel == "Channel 3") {	SelectChan = 3; }}}
		Stack.setPosition(SelectChan, 1, 1);	
	}	
	TrackCorrect = "Retrack";

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////// PART 2 ///////////////// PART 2 ///////////////// PART 2 ///////////////// PART 2 ///////////////// PART 2 ///////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

//===> 2.3     <========= Align - Start Looping =======================================================
	print ("   > > >  Cell : 1");
	for (cell = 1; cell < SelectRegion+1; cell++) {
		FileNamecrop = FileName+"_Cell"+cell;
		ROIstart = roiManager("count");
		if (TrackCorrect != "Retrack") {
			print ("   > > >  Cell : " + cell);
		}
		selectWindow(FileName);
		if (cell == 1) {
		}

//===> 2.4     <========= Align - Tracking =======================================================
		if (Track == "Yes") {
			selectWindow(FileName);
//===> 2.4.1a     <========= Align - Tracking - Follow =======================================================
			if (SelectMethod == "Follow") { 	// Method : [Follow]
				print ("          - Selection Speed : " + FPS + " FPS");
				setTool("hand");
				Stack.getDimensions(W, H, C, S, F);
				print ("          - FPS for tracking spots : ", FPS, " ( total frames : ", frames, " )");
				Stack.setPosition(SelectChan, 1, 1);
				if (cell == 1) {	if (TrackCorrect != "Retrack") {
					run("In [+]");	run("In [+]");
				}}
				roiManager("Show None");
				if (frames == 1) {
					close(FileName);
					exit("The macro will be terminated, only one frame available");
				}
				waitForUser("START FOLLOWING", "Cell >> " + cell + " << \n \n \n \n READY FOR FOLLOWING TS # " + cell +"  ? \n \n                                                                 Starts in 3 sec");	
				Stack.setPosition(SelectChan, 1, 1);
				print ("              ! ! ! Starting in 3 sec..."); 	wait(1000.);
				print ("              ! ! ! Starting in 2 sec..."); 	wait(1000.) ;
				print ("              ! ! ! Starting in 1 sec..."); 	wait(1000.);

				for (i = 1; i < Frames +1; i++) {
					setSlice(i);
					Stack.setPosition(1, 1, i);
					wait(1000/FPS);
					getCursorLoc(x, y, z, flags);
					makeOval(x-Radius, y-Radius, Radius*2, Radius*2);
					roiManager("add");
					ROIr = Roi.getName;
					print("             ~ ROI : 1 , Name : ", ROIr);
				}
			}
//===> 2.4.1b     <========= Align - Tracking - Click =======================================================
			if (SelectMethod == "Click") {		// Method : [Click]
				RadiusPerc = round(( Radius / width ) * 100);
				print ("          - Radius for selection : " + Radius * 2 + "   ( with selection in center  [", RadiusPerc, "% of image ] )");
				Stack.setPosition(1, 1, 1);			
				setTool("point");
				ROIlast = roiManager("count") -1;
				roiManager("Show All without labels");
				run("Point Tool...", "type=Circle color="+ROIcolor+" size=Large add label");
				// write in log file what size of ROI has been predefined and size of image
				waitForUser("START SELECTING", "                                          Cell >> " + cell + " << \n \n \n Slide through time and (re)define ROIs. Press ok when ready !!!");
				// create ROI's for all frames
				ROIlast = roiManager("count") -1;
			// Reselect Points
				if (ROIlast  == ROIstart - 1) { // start of restart selecting
					waitForUser("RESELECT", "\n \n    Seems like you haven't created ROIs. \n \n    Add ROIs. \n    Press ok when ready !!!   \n \n                                              Cell >> " + cell + " << ");					
					print ("          - Reselect cells, no ROIs were clicked");
				} // end of restart selecting

				ROIend = roiManager("count");
				RadiusHalf = Radius / 2;
			// Single ROI
				if (ROIlast - 1 == ROIstart - 1) { // start of single ROI
					print ("          - Amount of ROIs Selected : 1");
					roiManager("select", ROIlast);
					ROIr = Roi.getName;
					print("             ~ ROI : 1 , Name : ", ROIr);
					Stack.getPosition(chanROI, sliROI, frameROI);
					RoiSelName = Roi.getName;
					Roi.getCoordinates(x, y);
					for (i=0; i<x.length; i++) {
						setPixel(x[i], y[i]);
						xpos = round(x[i]);	
						ypos = round(y[i]);
					}					
					for (fc = 1; fc < frames+1; fc++) {
						Stack.setPosition(chanROI, sliROI, fc);
						makeOval(xpos-RadiusHalf, ypos-RadiusHalf, Radius, Radius);
						roiManager("Add", ROIcolor);
						roiRename = roiManager("count") - 1;
						roiManager("select", roiRename);
						roiManager("rename", fc + "-" + ypos + "-" + xpos + "_(R:" + ROIlast + "/S: " + frameROI+ ")");
					} // end of loops of frames
					roiManager("select", ROIend -1);
					roiManager("delete");
				} // end of single ROI
			// Multiple ROIs
				if (ROIlast - 1 > ROIstart - 1) { // start of multiple ROIs
					print ("          - Amount of ROIs Selected : ", ROIlast - ROIstart +1);					
					print ("       * Clicked ROIs : ");
					for (ro = ROIstart; ro < ROIlast - ROIstart +1; ro++) {
						roiManager("select", ro);
						ROIclick = Roi.getName;
						print("          - ROI : ", ro+1, ", Name : ", ROIclick);
					}			
					getDimensions(width, height, channels, slices, frames);
					roiManager("select", ROIend -1);
					Stack.getPosition(channel, slice, frame);
					if (frame < frames) {
						roiManager("select", ROIend -1);
						Stack.setPosition(channel, slice, frames);
						roiManager("add");
					}
					ROIend = roiManager("count")-1;
					ROIclick = ROIend - ROIstart;
					RoiSel = ROIstart +1;
					for (fc = 1; fc < frames+1; fc++) {
						roiManager("select", RoiSel);
						ROIr = Roi.getName;
						Stack.getPosition(channelROI, sliceROI, frameROI);
						Stack.setPosition(1, 1, fc);
						if (fc == frameROI) {	
							RoiSel = RoiSel;
						}
						if (fc < frameROI) {
							RoiSel = RoiSel - 1;
							if (RoiSel < 0) {	RoiSel = 0;	}	
						}
						roiManager("select", RoiSel);
						Stack.getPosition(chanROI, sliROI, fraROI);
						RoiSelName = Roi.getName;
						Roi.getCoordinates(x, y);
						for (i=0; i<x.length; i++) {
							setPixel(x[i], y[i]);
							xpos = round(x[i]);	
							ypos = round(y[i]);
						}
						if (fc <= fraROI ) {
							RoiSel = RoiSel;
						}
						else {
							RoiSel = RoiSel + 1;
							roiManager("select", RoiSel);							
							RoiSelName = Roi.getName;
							Roi.getCoordinates(x, y);
							for (i=0; i<x.length; i++) {
								setPixel(x[i], y[i]);
								xpos = round(x[i]);	
								ypos = round(y[i]);
							}
						}
						roiManager("select", RoiSel);
						Stack.setPosition(chanROI, sliROI, fc);
						makeOval(xpos-RadiusHalf, ypos-RadiusHalf, Radius, Radius);
						roiManager("Add", ROIcolor);
						roiRename = roiManager("count") - 1;
						roiManager("select", roiRename);
						roiManager("rename", fc + "-" + ypos + "-" + xpos + "_(R:" + RoiSel + "/S: " + fraROI+ ")");
					}
					for (rd = 0; rd < ROIend+1; rd++) {
						roiManager("select", 0);
						roiManager("delete");
					} // end of delete clicked ROIs
				} // end of multiple ROIs
			} // end of method "Click"

//===> 2.4.2     <========= Align - Tracking - Create ROI =======================================================
			ROIend = roiManager("count");
			ROIcell = ROIend - ROIstart;
			
		// Play through selected ROI's over image (to check if tracked well)
			if (CheckNot == true) {
				TrackCorrect = "Yes";
			}
			if (CheckEach == true) {
				// waitForUser("Your selected point will be played");
				wait(1000);
				fps = FPS * 4;
				print ("       * Rename ROIs : ");
				for (f = 0; f < ROIcell; f++) {
					roiManager("Select", f);
					roiManager("Set Color", ROIcolor);	roiManager("Set Line Width", 3);
					ROIname = Roi.getName;
					Fr = f + 1;
					roiManager("rename", "Im:" + FileName + "Cell:" + cell + "_Nr:" + Fr + "_" + ROIname);
					print("          - Cell : ", cell, ", Frame : ",  Fr,  " = Position : " + ROIname);
					wait(fps);
				}
				Dialog.create(""+title+", "+version+" ::-:: View Tracks ");
				items = newArray("Yes", "Replay", "Retrack", "Adjust Points");
				Dialog.addRadioButtonGroup("Is tracking performed correct ? ", items, 1, 4, "Yes");
				Dialog.addMessage("");
				Dialog.addMessage(" Yes                    = Continue to next step \n Replay               = Replay the performed tracking (to check)  \n Retrack              = Perform tracking again of cell "+cell+" \n Adjust Points     = Adjust a few points within the whole selection of cell "+cell+"");
				Dialog.addMessage("");
				Dialog.show();
				TrackCorrect = Dialog.getRadioButton();
//===> 2.4.3     <========= Align - Tracking - Replay ROI =======================================================
				if (TrackCorrect == "Replay") {
					fps = FPS * 4;
					roiManager("Show None");
					for (f = 0; f < ROIcell; f++) {
						roiManager("Select", f);
						wait(fps);
					}
					Dialog.create(""+title+", "+version+" ::-:: View Tracks ");
					items = newArray("Yes", "Retrack", "Adjust Points");
					Dialog.addRadioButtonGroup("Is tracking performed correct ? ", items, 1, 4, "Yes");
					Dialog.addMessage("");
					Dialog.addMessage("If selected [no], you can draw cell " + cell +" again");
					Dialog.addMessage("");
					Dialog.show();
					TrackCorrect = Dialog.getRadioButton();					
				} // end of replay ROIs
//===> 2.4.4     <========= Align - Tracking - Adjust ROI points =======================================================
				if (TrackCorrect == "Adjust Points") {
					run("ROI Manager...");
					setTool("rectangle");
					roiManager("select", 0);
					run("In [+]");
					waitForUser("Rearrange ROI's", "\n\n Rearrange desired ROI's : \n      (1) Select positions to correct in ROI manager, \n      (2) move ROI to desired location, \n      (3) press Update in ROI manager. \n\n");
					TrackCorrect = "Yes";
				}
//===> 2.4.4     <========= Align - Tracking - Save ROIs =======================================================
				if (TrackCorrect == "Yes") {
					// Deselect ROI's to save as list file
					roiManager("save", filedir + FileName + "_ROI" + "_Cell" + cell + ".zip");
					ROIselect = "Cell:" + cell;
					nR = roiManager("Count");	arrayROI_Cell = newArray();
					for (i=0; i<nR; i++) {		roiManager("Select", i);	rName = Roi.getName(); 
					if (indexOf(rName, ROIselect) >=0) {} else { 	roiManager("Select", i);	arrayROI_Cell = Array.concat(arrayROI_Cell,i); 	}}
					roiManager("select", arrayROI_Cell);
					if (arrayROI_Cell.length != 0) {
						roiManager("delete");
					}
					roiManager("Deselect");
					roiManager("List");
					saveAs("Results", filedir + FileName + "_ROI" + "_Cell" + cell + ".txt");
					print ("          - ROI Text File Saved : " + FileName + "_ROI" + "_Cell" + cell + ".txt");
					close(FileName + "_ROI" + "_Cell" + cell + ".txt");
					roiFin = roiManager("count");
					if (roiFin != 0) {
						roiManager("deselect");		roiManager("delete");
					}
				} // end of track correction
				if (TrackCorrect == "Retrack") {
					Cell = cell;
					cell = cell - 1;
					print ("   > > >  Cell : " + Cell + "     ( = RETRACKING ) ");
					roiFin = roiManager("count");
					if (roiFin != 0) {
						roiManager("deselect");		roiManager("delete");
					}
				}
			} // end of trackview yes		
		} // end of "Track"

//===> 2.5     <========= Align - Crop =======================================================	
//===> 2.5.1     <========= Align - Crop - Create Crop Image =======================================================
		if (TrackCorrect == "Yes") {
			if (Crop == "Yes") {
				newImage(FileNamecrop, Bit+" color-mode", CropWidth, CropHeight, Channels, Slices, Frames);
				for (C = 1; C < Channels +1; C++) {
					setSlice(C);
					if (C == 1) {	run(ColorChannel1); }
					if (C == 2) {	run(ColorChannel2); Stack.setDisplayMode("composite");	}
					if (C == 3) {	run(ColorChannel3); Stack.setDisplayMode("composite");	}
				}
				// create "emtpty" border around image
				selectWindow(FileName);
				getDimensions(Width, Height, channels, slices, frames);

				if (cell == 1) {			
					WidthCanvas = Width + CropWidth;		HeightCanvas = Height + CropHeight;		// to create a broader boundry, so that spots close to edge will still be copied
					HalfWidthCrop = CropWidth / 2;			HalfHeightCanvas = CropHeight / 2;		// to add up for ROI location
					run("Select None");
					run("Duplicate...", "title=SelectImage duplicate");
					run("Canvas Size...", "width="+WidthCanvas+" height="+HeightCanvas+" position=Center");
				} // end of drawing border
						
				roiCellStart = roiManager("count") - Frames;

//===> 2.5.2     <========= Align - Crop - Create Region =======================================================
				for (F = 1; F < Frames +1; F++) {
					selectWindow("SelectImage");
					roiManager("select", roiCellStart);
					Roi.getBounds(ROIx, ROIy, ROIwidth, ROIheight);		// get dimensions of spot
					Stack.setPosition(SelectChan, 1, F);
					makeRectangle(ROIx, ROIy, CropWidth, CropHeight);
					HalfRad = Radius /2;		// calculates center of defined spot (from width)
					WidthHalf = CropWidth / 2;				HeightHalf = CropHeight / 2;		// calculates half of crop image (width and height)
					Xpos = ROIx - (2 * WidthHalf) ;			Ypos = ROIy - (2 * HeightHalf);		// calculates start position (width and height)
					Xbord =	Xpos + (2 * WidthHalf);			Ybord =	Ypos + (2 * HeightHalf);		// calculates end position from start (width and height)
					ratio = Width / Height;
					roiManager("add");
					roicur = roiCellStart + Frames;
					roiManager("select", roicur);
					ROIc = Roi.getName;
					
					roiManager("select", roicur);
					roiManager("rename", "Cell"+cell+"_Crop_"+F+"_("+ROIc+")_{Border"+WidthHalf+"}");
					print("            ~ ROI drawn : ", "Crop_"+F+"_("+ROIc+")_{Border"+WidthHalf+"}");
					for (C = 1; C < Channels +1; C++) {
						if (slices == 1) {	
							selectWindow("SelectImage");
							Stack.setPosition(C, 1, F);
							roiManager("select", roicur);
							run("Copy");
							selectWindow(FileNamecrop);	
							Stack.setPosition(C, 1, F);
							run("Select All");
							run("Paste");
							run("Enhance Contrast", "saturated=0.35");
						} // end of else statement (if slices = 1)
						else {
							for (S = 1; S < Slices +1; S++) {
								selectWindow("SelectImage");
								Stack.setPosition(C, S, F);
								roiManager("select", roicur);
								run("Copy");
								selectWindow(FileNamecrop);	
								Stack.setPosition(C, S, F);
								run("Select All");
								run("Paste");
								run("Enhance Contrast", "saturated=0.35");
							} 
						} // end of else statement (if slices > 1)
					} // end of run through channels
					roiCellStart = roiCellStart + 1;
				} // end of run through channels frames to copy
				saveAs("Tiff", filedir + FileNamecrop + "_Crop" + CropWidth + "_max");
				print("          - Crop image saved as : ", FileNamecrop + "_Crop" + CropWidth + "_max", "tiff");
				rename(FileNamecrop);
			} // end of "Crop"

//===> 2.6     <========= Align - Segment =======================================================
//===> 2.6.1     <========= Align - Segment - Dialog =======================================================
			if (Sum == "Yes") {
				selectWindow(FileNamecrop);
				getDimensions(widthCrop, heightCrop, channelsCrop, slicesCrop, framesCrop);
				if (framesCrop > 1) {
					run("Z Project...", "projection=[Sum Slices]");
					rename(FileNamecrop+"_sum");
				}
				setTool(SelectTool);
				selectWindow(FileNamecrop+"_sum");
				for (c = 0; c < Channels + 1; c++) {
					Stack.setPosition(c, 1, 1);
					run("Enhance Contrast", "saturated=0.35");
				}
				if (cell == 1) {
					Dialog.create(""+title+", "+version+" ::-:: Background correction");
					Dialog.addNumber("Segment background subtraction : ", 100, 0, 5,"");
					Dialog.addMessage("                     0 = no background subtraction ");
					Dialog.addMessage("                   10 = heavy ");
					Dialog.addMessage("                 100 = minimal ");
					Dialog.show();
					Bkgr = Dialog.getNumber();
				}
				if (Bkgr != 0) {
					run("Subtract Background...", "rolling="+Bkgr+"");
					getDimensions(widthCrop, heightCrop, channelsCrop, slicesCrop, framesCrop);
					run("Select None");
					for (c = 1; c < channels +1; c++) {
						setSlice(c);
						run("Enhance Contrast", "saturated=0.35");
					}
				}
//===> 2.6.2     <========= Align - Segment - Create Segment Image =======================================================
				waitForUser("SEGMENT IMAGE", "Draw region of transcription site, for segmentation image ");
				roiManager("Add");
				ROIsum = roiManager("count") - 1;
				roiManager("select", ROIsum);
				roiManager("rename", "Cell"+cell+"_Segment");
				FileNamecropSum = FileNamecrop+SumName;
				newImage(FileNamecropSum, Bit+" grayscale-mode", CropWidth, CropHeight, 1, 1, 1);
				roiManager("select", ROIsum);
				run("Add...", "value=1");
				run("Enhance Contrast", "saturated=0.35");
				saveAs("Tiff", filedir + FileName + "_Cell" + cell + "_" +SumName);
				print ("          - Segment image saved : " + FileName + "_Cell" + cell + "_" +SumName + ".tiff");
				rename(FileNamecrop+SumName);		
			} // end of "Sum (Segment)"

			if (Sum == "Yes") {
				close(FileNamecrop+"_sum");
				close(FileNamecrop+SumName);
			}
			if (Crop == "Yes") {
				close(FileNamecrop);
			}
		} // end of track correct
	} // end of cell loop

//===> 3     <========= Save =======================================================
	print ("- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -");
	print ("3 # Save & Close :");
//===> 3.1     <========= Save - Dialog =======================================================
	run("Scale to Fit");
	Dialog.create(""+title+", "+version+" ::-:: Save / End ");
		items = newArray("Yes", "No");
	Dialog.addRadioButtonGroup("Save Log File : ", items, 1, 2, "Yes");		
	Dialog.addMessage("");
	if (CheckEnd != true) {
		items = newArray("Yes", "No");
		item = "Yes";
	}
	if (CheckEnd == true) {
		items = newArray("Yes", "Finished", "Check All Cells");
		item = "Check All Cells";
	}
	Dialog.addRadioButtonGroup("Are you ready to close all images/data : ", items, 1, 2, item);		
	Dialog.addMessage("");
	Dialog.show();
	SaveLog = Dialog.getRadioButton();
	Final = Dialog.getRadioButton();

	if (Final == "Check All Cells") { Check = "Yes (.png file saved)"; } else { Check = "No"; }
	print ("          - Check all cells at end : ", Check);
	if (Final == "Yes") {
		print ("          - Close images :            ", Final);
	}

//===> 3.2     <========= Save - Close =======================================================
	if (SaveImage == true) {	Saveimage = "Yes"; } else {	Saveimage = "No"; }
		print ("          - Save Image Overlay : ", Saveimage);	
	if (SaveImage == true) {
		Final = "Check All Cells";
	}
	if (Final == "Check All Cells") {
		getDimensions(width, height, channels, slices, frames);
		run("Canvas Size...", "width="+width+" height="+height+" position=Center");
		list = getFileList(filedir);
		for ( i=0; i<list.length; i++ ) { 
		    if(endsWith(list[i], ".zip")){
		    	open(filedir + list[i] );
		    }
		}
		roiManager("Deselect");		
		roiManager("Set Color", ROIcolor);	roiManager("Set Line Width", 1);
		roiManager("Show All");		
		if (SaveImage == true) {
			Deletelist = getFileList(filedir);
			for ( i=0; i<Deletelist.length; i++ ) {
			    if(endsWith(Deletelist[i], ".zip")){
			    	roiManager("Open", filedir + Deletelist[i]);
			    	File.delete(filedir + Deletelist[i]);
			    }
			}
			roiManager("Show None");
			
			if (SelectMethod == "Follow") {
				Radius = width/ 100;	RadiusHalf = Radius / 2;
			}
			ROIoverlay = roiManager("count");
			for (rl = 0; rl < ROIoverlay; rl++) {
				roiManager("Select", rl);
				Roi.getCoordinates(x, y);
				for (i=0; i<x.length; i++) {
					setPixel(x[i], y[i]);
					xpos = round(x[i]);	
					ypos = round(y[i]);
				}
				Stack.setPosition(1, 1, 1);
				makeOval(xpos-RadiusHalf, ypos-RadiusHalf, Radius, Radius);
				roiManager("Update");
			}
			roiManager("deselect");
			run("From ROI Manager");
			PosW = 10000 / width;		PosH = 25000 / height; 		fontsize = 18000 / height;
			if (bitDepth() == 8) { TextColor = 255; }	if (bitDepth() == 16) { TextColor = 65535; }	if (bitDepth() == 32) { TextColor = 1; }		
			setFont("Arial", fontsize, "Not");	setColor(TextColor);
			for (c = 1; c < channels+1; c++) {
				Stack.setPosition(c, 1, 1);
				drawString ("File: " + FileName, PosW, PosH);
				drawString ("(Ch: " + channels + ", Fr: " + frames + ", ROI: " + SelectRegion + "{f:1}) [" + SelectMethod + "-"+DateString+"] ", PosW, PosH+PosH);
				print ("          - Overlay Image Saved : " + FileName + "_overlay[" + DateString + "-"+TimeString+"].png");
				print ("              - Channels : " + channels + ", Frames: " + frames + "  (current frame in image : 1)");
				print ("              - ROIs created : " + SelectRegion + ", Selection Method : " + SelectMethod);
				print ("              - Date created : " + DateString);
			}
			if (channels > 1) {
				Stack.setDisplayMode("composite");
			}
			run("From ROI Manager");
			roiManager("Show All without labels");
			saveAs("PNG", filedir + FileName + "_overlay[" + DateString + "-"+TimeString+"].png");
		} // end of save image = true
		if (Final != "Finished") {
			Dialog.create(""+title+", "+version+" ::-:: End ");
				items = newArray("Yes", "No");
			Dialog.addRadioButtonGroup("Are you ready to close all images/data : ", items, 1, 2, "Yes");		
			Dialog.show();
			Final = Dialog.getRadioButton();
		} // end of final not finished
	} // end of final check all cells
	print ("       * Time Spent on Macro :  ", d2s(((getTime()-StartTime)/60000),2), " min");

//===> 3.3     <========= Save - Log file =======================================================
	if (SaveLog == "Yes") {
		if (Final == "Yes") {
			print ("          - Close images :            ", Final);
		}
		selectWindow("Log");
		print ("          - Log file saved : " + FileName + "_Log_[" + MACRO + "_" + DateString + "_" + TimeString + "].txt");
		saveAs("Text", filedir + FileName + "_Log_[" + MACRO + "_" + DateString + "_" + TimeString + "].txt");
	} // end of save log file

	
//===> 3.4     <========= Save - Close =======================================================	
	if (Final == "Yes") {
		run("Close All");
		close("Log");
		roiManager("deselect");
		roiManager("delete");
	}
} // end of macro

