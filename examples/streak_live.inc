

TYPE WCS
    crval1 AS DOUBLE
    crval2 AS DOUBLE
    crpix1 AS DOUBLE
    crpix2 AS DOUBLE
    cd11   AS DOUBLE
    cd12   AS DOUBLE
    cd21   AS DOUBLE
    cd22   AS DOUBLE
END TYPE


TYPE Vec3
    x AS DOUBLE
    y AS DOUBLE
    z AS DOUBLE
END TYPE

TYPE GoodingObs
    JD  AS DOUBLE
    RA  AS DOUBLE
    Dec AS DOUBLE
    los AS Vec3
    R   AS Vec3
    ExpRange AS DOUBLE
END TYPE

GLOBAL gObs() AS GoodingObs


SUB StreakTrackCheck
    STATIC SNorad AS STRING
    LOCAL NewStreak AS LONG


    IF ischkASCOMContinuous=1 THEN EXIT SUB
    IF gStreakEnable<>1 THEN EXIT SUB
    IF sNorad<>CNorad THEN
        NewStreak=1
        sNorad=cNorad
        gStreakTotalCnt=0
        gStreakGoodCnt=0
    ELSE
        NewStreak=0
    END IF

    IF isStreakDialog AND NewStreak THEN

         DIALOG END hStreakDlg
    END IF

    DIALOG NEW 0, "Streak Imaging: ["+cNorad+"] "+cSatelliteName, , ,%MainDlgWidth,%MainDlgHeight, _
                            %WS_SYSMENU OR %WS_MINIMIZEBOX OR %WS_MAXIMIZEBOX OR %WS_THICKFRAME  TO hStreakDlg

   ' DIALOG NEW hMainDlg, "Streak Imaging: ["+cNorad+"] "+cSatelliteName, , , %MainDlgWidth,%MainDlgHeight,%WS_SYSMENU OR %WS_MINIMIZEBOX  ,  TO hStreakDlg
    DIALOG SHOW MODELESS hStreakDlg CALL StreakProc
    DIALOG SET ICON hStreakDlg, "#10"


END SUB

CALLBACK FUNCTION StreakProc()
    LOCAL lpLstPasses24hr AS nmlvcustomdraw PTR

    SELECT CASE CB.MSG
        CASE %WM_INITDIALOG
            LOCAL ViewFont AS STRING, hViewFont1 AS LONG
               ViewFont=ListFontName


            CONTROL ADD LISTVIEW,hStreakDlg,%lstStreak," " ,10,10,638,370,%WS_TABSTOP OR %LVS_REPORT OR %LVS_NOSORTHEADER OR %LVS_SHOWSELALWAYS OR _
                                                   %LVS_SINGLESEL ,%WS_EX_STATICEDGE OR %WS_EX_LEFT



            CONTROL SET FONT hStreakDlg,%lstStreak, hViewFont1
            LISTVIEW SET STYLEXX hStreakDlg,%lstStreak, %LVS_EX_DOUBLEBUFFER OR %LVS_EX_FULLROWSELECT 'OR %LVS_EX_GRIDLINES

            CONTROL HANDLE hStreakDlg,%lstStreak TO ghListviewZ
            ghHeaderZ = SendMessage(ghListviewZ, %LVM_GETHEADER, 0, 0)
            OldSubClassProcZ = SetWindowLong(GetDlgItem(hStreakDlg,%lstStreak ),  %GWL_WNDPROC, CODEPTR(ListViewProc))

            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 1," Reject",18,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 2," Comment",115,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 3," Image",84,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 4," Date/Time",72,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 5," Start X",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 6," Start Y",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 7," End X",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 8," End Y",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 9," Mid X",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 10," Mid Y",35,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 11," Peak SNR",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 12," Mean SNR",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 13," Elongation",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 14," Obs Length",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 15," Exp Length",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 16," Start RA",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 17," Start Dec",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 18," End RA",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 19," End Dec",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 20," Mid RA",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 21," Mid Dec",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 22," Expected RA",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 23," Expected Dec",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 24," Separation (Arcmin)",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 25," Exposure",45,0
            LISTVIEW INSERT COLUMN hStreakDlg,%lstStreak, 26," Gain",45,0
            gStreakLstCnt=0


            DIALOG SET CLIENT hStreakDlg, gStreakDialogWidth, gStreakDialogHeight
            DIALOG SET LOC hStreakDlg, gStreakDialogX, gStreakDialogY
            IF gNightMode>0 THEN
                IF CheckDarkModeAuto=0 THEN
                    DarkModeSet gNightMode
                END IF
            END IF

       CASE %WM_EXITSIZEMOVE
            LOCAL w AS LONG, h AS LONG

            'DIALOG GET CLIENT CB.HNDL TO w, h
            'CONTROL SET SIZE CB.HNDL, %lstStreak, w, h

            DIALOG GET CLIENT hStreakDlg TO gStreakDialogWidth, gStreakDialogHeight
            DIALOG GET LOC hStreakDlg TO gStreakDialogX, gStreakDialogY

        CASE %WM_SIZE
             'LOCAL w AS LONG, h AS LONG

             DIALOG GET CLIENT CB.HNDL TO w, h
             CONTROL SET SIZE CB.HNDL, %lstStreak, w-20, h-20

       CASE %WM_NOTIFY   'time to draw ownerdrawn button
            SELECT CASE CB.WPARAM

                IF CB.NMID=%lstStreak THEN
                SELECT CASE CB.NMCODE
                    CASE %NM_CUSTOMDRAW
                        lpLstPasses24hr=CB.LPARAM
                        SELECT CASE @lpLstPasses24hr.nmcd.dwDrawStage
                            CASE %CDDS_PrePaint, %CDDS_ItemPrePaint
                                FUNCTION=%CDRF_NotifyItemDraw
                            CASE %CDDS_ItemPrepaint OR %CDDS_subitem
                                IF @lpLstPasses24hr.nmcd.dwItemSpec MOD 2 =1 THEN
                                    @lpLstPasses24hr.clrTextBk=lstBkColor1
                                    @lpLstPasses24hr.clrText=lstFgColor1
                                ELSE
                                    @lpLstPasses24hr.clrTextBk=lstBkColor2
                                    @lpLstPasses24hr.clrText=lstFgColor2
                                END IF
                        END SELECT
                END SELECT
            END IF

              ' CASE %bnSolveCancel
               '    DrawButton GetDlgItem(CB.HNDL, CBWPARAM), CB.LPARAM, gBNBkColor, gBnTextColor
            END SELECT


       CASE %WM_COMMAND
            SELECT CASE CB.CTL

            END SELECT


        CASE  %WM_MOVE, %WM_SIZE


    END SELECT
END FUNCTION

FUNCTION isStreakDialog AS LONG
    LOCAL x1, y1 AS LONG

    DIALOG GET LOC hStreakDlg TO x1, y1
    IF x1+y1<>0 THEN
        FUNCTION=1
    ELSE
        FUNCTION=0
    END IF

END FUNCTION


FUNCTION StreakJD(StreakDT AS STRING) AS DOUBLE
    LOCAL yr AS LONG,yrd AS DOUBLE,mo AS LONG, da AS LONG
    LOCAL hr AS LONG, mi AS LONG,se AS LONG, hu AS LONG

    yr =VAL(MID$(StreakDT,1,4))
    mo =VAL(MID$(StreakDT,6,2))
    da =VAL(MID$(StreakDT,9,2))

    hr=VAL(MID$(StreakDT,12,2))
    mi=VAL(MID$(StreakDT,15,2))
    se=VAL(MID$(StreakDT,18,2))
    hu=VAL(MID$(StreakDT,21,3))
    yrd=yr
    FUNCTION=JulianDateOfYear(yrd)+DayOfYear(yr,mo,da)+FractionOfDay(hr,mi,se,hu)

END FUNCTION




FUNCTION StreakProcess(ImagesPath AS STRING,FitsFile AS STRING, CSat AS LONG) AS LONG
    LOCAL siWidth AS LONG, siHeight AS LONG , sImg() AS INTEGER
    LOCAL ireturn, istep AS LONG
    LOCAL pImg AS INTEGER PTR
    LOCAL background, sigma AS DOUBLE, mask() AS BYTE
    LOCAL x, y AS LONG
    LOCAL labels() AS LONG, nComponents AS LONG,centroids() AS DOUBLE, areas() AS LONG, bbox() AS LONG
    LOCAL MinArea AS LONG, keepMask() AS BYTE
    LOCAL startX, startY,endX, endY, midx, midy AS DOUBLE
    LOCAL largestlabel, edgeFlag, hFile AS LONG
    LOCAL  peakSNR, meanSNR, elongation, ObservedLength, ExpectedLength AS DOUBLE
    LOCAL comment, idateTime, FitsFile2 AS STRING
    LOCAL Solved AS EQCoordsType, WCSFile AS STRING, iWCS AS WCS, SolvedOk AS LONG
    LOCAL MidRA,MidDec, StartRA, StartDec, EndRA, EndDec, iExposure, iGain, SiteLat, SiteLong,  SiteElev AS DOUBLE
    LOCAL NORAD AS STRING, ExpectedRA, ExpectedDec, MidSeparation, ExpectedRange AS DOUBLE


    comment=""
    siWidth=0
    siHeight=0
    StartX=0
    StartY=0
    EndX=0
    EndY=0
    MidX=0
    MidY=0
    peakSNR=0
    meanSNR=0
    elongation=0
    ObservedLength=0
    edgeFlag=-1


    iReturn = FITSLoadInt16(ImagesPath+FitsFile, sImg(), siWidth,siHeight,iDateTime, iExposure, iGain, SiteLat, SiteLong,  SiteElev)



    IF iReturn<>-1 THEN
        IF iReturn=-2 THEN comment ="Image not 16bit"
        IF iReturn=0 THEN comment="Error loading image"
        GOTO SaveAnalysis
    END IF

    iStep=4
    pImg  = VARPTR(sImg(0,0))
    StreakFastBackgroundMAD (pImg,siWidth, siHeight,istep, background, sigma)

    'create mask
    REDIM mask( 0 TO SiWidth-1,0 TO siHeight-1 ) AS BYTE
    FOR y = 0 TO siHeight-1
        FOR x = 0 TO siWidth-1
            IF sImg(x, y) > background + 1 * sigma THEN
                mask(x, y) = 1
            ELSE
                mask(x, y) = 0
            END IF
        NEXT
    NEXT

    FOR y = 1 TO siHeight-2
        FOR x = 1 TO siWidth-2
            IF mask(x,y) THEN
                LOCAL n AS LONG
                n = 0
                IF mask(x-1,y) THEN INCR n
                IF mask(x+1,y) THEN INCR n
                IF mask(x,y-1) THEN INCR n
                IF mask(x,y+1) THEN INCR n
                IF mask(x-1,y-1) THEN INCR n
                IF mask(x+1,y-1) THEN INCR n
                IF mask(x-1,y+1) THEN INCR n
                IF mask(x+1,y+1) THEN INCR n
                IF n < 2 THEN mask(x,y) = 0
            END IF
        NEXT
    NEXT


    'label Components
    minArea=200
    StreakLabelComponents mask(), siWidth, siHeight, sImg(), _
                      background, sigma, minArea, _
                      startX, startY, endX, endY, _
                      midX, midY, peakSNR, meanSNR, _
                      elongation, ObservedLength

    edgeFlag=StreakOnEdge(siWidth, siHeight, StartX, StartY, EndX, EndY)
    IF elongation<gStreakMinElongation THEN  'probably a star
        Comment="No streak found"
        StartX=0
        StartY=0
        EndX=0
        EndY=0
        MidX=0
        MidY=0
        peakSNR=0
        meanSNR=0
        'ObservedLength=0
        edgeFlag=-1
        GOTO SaveAnalysis
    END IF

    IF EdgeFlag=1 THEN Comment="Streak on the edge"

    IF peakSNR<gStreakMinPeakSNR THEN 'peak SNR too low
         Comment="Signal too low"
         edgeFlag=2
         GOTO SaveAnalysis
    END IF

    IF ObservedLength < gStreakMinLength THEN
         Comment="Streak <"+FORMAT$(gStreakMinLength) +" pixels"
         edgeFlag=3
         GOTO SaveAnalysis
    END IF



    FitsFile2=REMOVE$(FitsFile,".fits")
    WCSFile=""
    IF edgeFlag=0 THEN  'got a good streak so solve it
        SolvedOk= SolveFits(1 , ImagesPath, FitsFile2,Solved, WCSFile)
        IF SolvedOK=0 THEN
              SolvedOk= SolveFits(2 , ImagesPath, FitsFile2,Solved, WCSFile)
        END IF

        IF SolvedOk=1 THEN
            Writeaudit 1,"Solved Ok, File="+WCSFILE
            IF WCSFileLoad(WCSFile,iWCS) THEN
                StreakPixelToRaDec midX,midY,iWCS,MidRA,MidDec
                StreakPixelToRaDec startX,startY,iWCS,startRA,startDec
                StreakPixelToRaDec endX,endY,iWCS,endRA,endDec
            ELSE
                Comment="Could not platesolve image"
                edgeFlag=4
                GOTO SaveAnalysis
            END IF
        ELSE
           Comment="Could not platesolve image"
           edgeFlag=4
           GOTO SaveAnalysis
        END IF
    END IF

    SaveAnalysis:

    NORAD=MID$(FitsFile,8,INSTR(8,FitsFile,"_")-8)
    IF SolvedOk=1 THEN
       StreakDirection iDateTime, VAL(NORAD), StartRA, StartDec, EndRa, EndDec
       StreakExpectedPos VAL(NORAD), iDateTime, iExposure, ExpectedRA, ExpectedDec, ExpectedRange, ExpectedLength
       MidSeparation=SeparationArcMin(MidRA ,MidDec ,ExpectedRA,ExpectedDec)
    END IF

    IF isStreakDialog THEN
          INCR gStreakLstCnt
          LISTVIEW INSERT ITEM hStreakDlg,%lstStreak, gStreakLstCnt,0,TRIM$(STR$(edgeflag))
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,2 ,Comment
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,3 ,FitsFile
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,4 ,idatetime
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,5 ,FORMAT$(StartX,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,6 ,FORMAT$(StartY,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,7 ,FORMAT$(EndX,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,8 ,FORMAT$(EndY,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,9 ,FORMAT$(MidX,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,10 ,FORMAT$(MidY,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,11 ,FORMAT$(PeakSNR,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,12 ,FORMAT$(MeanSNR,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,13 ,FORMAT$(Elongation,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,14 ,FORMAT$(ObservedLength,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,15 ,FORMAT$(ExpectedLength,"#")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,16 ,FORMAT$(startRA,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,17 ,FORMAT$(startDec,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,18 ,FORMAT$(endRA,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,19 ,FORMAT$(endDec,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,20 ,FORMAT$(midRA,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,21 ,FORMAT$(midDec,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,22 ,FORMAT$(ExpectedRA,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,23 ,FORMAT$(ExpectedDec,"#.00000000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,24 ,FORMAT$(MidSeparation,"#.000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,25 ,FORMAT$(iexposure,"#.0000")
          LISTVIEW SET TEXT hStreakDlg,%lstStreak, gStreakLstCnt,26 ,FORMAT$(igain,"#")
    END IF

    INCR gStreakTotalCnt
    IF EdgeFlag=0 THEN INCR gStreakGoodCnt
    hFile=FREEFILE
    norad=TRIM$(norad_pc(csat))
    OPEN ImagesPath+Norad+".strk" FOR APPEND AS # hFile
    IF LOF(hfile)=0 THEN 'print header
        PRINT #hFile,"[VERSION]"
        PRINT #hFile,"Skytrack "+CHR$(9)+$Version
        PRINT #hFile,"[SITE]"
        PRINT #hFile,CHR$(9)+"Latitude(deg)"+_
                   CHR$(9)+"Longitude(deg, East=+)"+_
                   CHR$(9)+"Elevation(m)"
        PRINT #hFile,FORMAT$(site(CurrentSite).sLatitude,"#.0000000")+_
                   CHR$(9)+FORMAT$(site(CurrentSite).sLongitude,"#.0000000")+_
                   CHR$(9)+FORMAT$(site(CurrentSite).sElevation,"#.0")
        PRINT #hfile,"[TLE]"
        PRINT #hFile,"NORAD"+_
            CHR$(9)+"EpochYear"+_
            CHR$(9)+"EpochDay"+_
            CHR$(9)+"Incl"+_
            CHR$(9)+"RAAN"+_
            CHR$(9)+"ECC"+_
            CHR$(9)+"ARGP"+_
            CHR$(9)+"MA"+_
            CHR$(9)+"MM"+_
            CHR$(9)+"BSTAR"+_
            CHR$(9)+"MM1"+_
            CHR$(9)+"MM2"+_
            CHR$(9)+"EphemType"+_
            CHR$(9)+"Elset"+_
            CHR$(9)+"Rev"+_
            CHR$(9)+"Name"+_
            CHR$(9)+"Object ID"+_
            CHR$(9)+"Class"

        PRINT #hFile, norad_pc(csat)+_   ' <-- NORAD
                 CHR$(9)+MID$(SatelliteData(CSat,2),19,2)+_        ' EpochYear
                 CHR$(9)+MID$(SatelliteData(CSat,2),21,12)+_       ' EpochDay
                 CHR$(9)+MID$(SatelliteData(CSat,3),9,8)+_         ' Incl
                 CHR$(9)+MID$(SatelliteData(CSat,3),18,8)+_        ' RAAN
                 CHR$(9)+MID$(SatelliteData(CSat,3),27,7)+_        ' ECC
                 CHR$(9)+MID$(SatelliteData(CSat,3),35,8)+_        ' ARGP
                 CHR$(9)+MID$(SatelliteData(CSat,3),44,8)+_        ' MA
                 CHR$(9)+MID$(SatelliteData(CSat,3),53,11)+_       ' MM
                 CHR$(9)+MID$(SatelliteData(CSat,2),54,8)+_        ' BSTAR
                 CHR$(9)+MID$(SatelliteData(CSat,2),34,10)+_       ' MM1
                 CHR$(9)+MID$(SatelliteData(CSat,2),45,8)+_        ' MM2
                 CHR$(9)+MID$(SatelliteData(CSat,2),63,1)+_        ' EphemType
                 CHR$(9)+MID$(SatelliteData(CSat,2),65,4)+_        ' Elset
                 CHR$(9)+MID$(SatelliteData(CSat,3),64,5)+_        ' Rev
                 CHR$(9)+MID$(SatelliteData(CSat,1),1,24)+_        ' SatelliteName
                 CHR$(9)+LaunchYear_PC(CSat)+_                    ' Object ID
                 CHR$(9)+MID$(SatelliteData(CSat,2),8,1)           ' Classifcation


      PRINT #hfile,"[OBS]"
      PRINT #hFile,"Image"+_
                   CHR$(9)+"Date Time(UTC)"+_
                   CHR$(9)+"JD Midpoint"+_
                   CHR$(9)+"Start X Pixel"+_
                   CHR$(9)+"Start Y Pixel"+_
                   CHR$(9)+"End X Pixel"+_
                   CHR$(9)+"End Y Pixel"+_
                   CHR$(9)+"Mid X Pixel"+_
                   CHR$(9)+"Mid Y Pixel"+_
                   CHR$(9)+"Peak SNR"+_
                   CHR$(9)+"Mean SNR"+_
                   CHR$(9)+"Elongation"+_
                   CHR$(9)+"ObsLength"+_
                   CHR$(9)+"ExpLength"+_
                   CHR$(9)+"Reject"+_
                   CHR$(9)+"Mid RA"+_
                   CHR$(9)+"Mid Dec"+_
                   CHR$(9)+"Start RA"+_
                   CHR$(9)+"Start Dec"+_
                   CHR$(9)+"End RA"+_
                   CHR$(9)+"End Dec"+_
                   CHR$(9)+"Expected RA"+_
                   CHR$(9)+"Expected Dec"+_
                   CHR$(9)+"Expected Range"+_
                   CHR$(9)+"Exposure"+_
                   CHR$(9)+"Gain"+_
                   CHR$(9)+"Comment"
    END IF

    PRINT #hFile, fitsFile+_
        CHR$(9)+iDateTime+_
        CHR$(9)+FORMAT$(streakjd(idateTime) + ((iExposure / 2) / 86400##),"#.0000000000")+_
        CHR$(9)+FORMAT$(StartX,"#")+_
        CHR$(9)+FORMAT$(StartY,"#")+_
        CHR$(9)+FORMAT$(EndX,"#")+_
        CHR$(9)+FORMAT$(EndY,"#")+_
        CHR$(9)+FORMAT$(MidX,"#")+_
        CHR$(9)+FORMAT$(MidY,"#")+_
        CHR$(9)+FORMAT$(peakSNR,"#")+_
        CHR$(9)+FORMAT$(meanSNR,"#")+_
        CHR$(9)+FORMAT$(elongation,"#")+_
        CHR$(9)+FORMAT$(ObservedLength,"#")+_
        CHR$(9)+FORMAT$(ExpectedLength,"#")+_
        CHR$(9)+FORMAT$(edgeFlag,"#")+_
        CHR$(9)+FORMAT$(midRA,"#.00000000")+_
        CHR$(9)+FORMAT$(midDec,"#.00000000")+_
        CHR$(9)+FORMAT$(startRA,"#.00000000")+_
        CHR$(9)+FORMAT$(startDec,"#.00000000")+_
        CHR$(9)+FORMAT$(endRA,"#.00000000")+_
        CHR$(9)+FORMAT$(endDec,"#.00000000")+_
        CHR$(9)+FORMAT$(ExpectedRA,"#.00000000")+_
        CHR$(9)+FORMAT$(ExpectedDec,"#.00000000")+_
        CHR$(9)+FORMAT$(ExpectedRange,"#.000")+_
        CHR$(9)+FORMAT$(iExposure,"#.0000")+_
        CHR$(9)+FORMAT$(iGain,"#.0")+_
        CHR$(9)+Comment

     CLOSE # hFile

    DIALOG SET TEXT hStreakDlg, "Streak Imaging: ["+norad_pc(csat)+"] "+cSatelliteName+"   "+FORMAT$(gStreakGoodCnt)+"/"+FORMAT$(gStreakTotalCnt)

END FUNCTION



FUNCTION WCSFileLoad (WCSFile AS STRING, iWCS AS WCS) AS LONG
    LOCAL f AS LONG
    LOCAL dtxt AS STRING
    LOCAL dataline AS STRING

    IF DIR$(WCSFile) = "" THEN EXIT FUNCTION

    f = FREEFILE
    OPEN WCSFile FOR INPUT AS #f
    IF ERR<>0 THEN writeaudit 1,"Error opening WCS "+STR$(ERR)

    DO UNTIL EOF(f)
        LINE INPUT #f, dataline
        dtxt = UCASE$(TRIM$(dataline))

        IF INSTR(dtxt,"CRVAL1") THEN iWCS.crval1 = WCSGetValue(dtxt)
        IF INSTR(dtxt,"CRVAL2") THEN iWCS.crval2 = WCSGetValue(dtxt)

        IF INSTR(dtxt,"CRPIX1") THEN iWCS.crpix1 = WCSGetValue(dtxt)
        IF INSTR(dtxt,"CRPIX2") THEN iWCS.crpix2 = WCSGetValue(dtxt)

        IF INSTR(dtxt,"CD1_1")  THEN iWCS.cd11   = WCSGetValue(dtxt)
        IF INSTR(dtxt,"CD1_2")  THEN iWCS.cd12   = WCSGetValue(dtxt)

        IF INSTR(dtxt,"CD2_1")  THEN iWCS.cd21   = WCSGetValue(dtxt)
        IF INSTR(dtxt,"CD2_2")  THEN iWCS.cd22   = WCSGetValue(dtxt)

    LOOP

    CLOSE #f

    IF iWCS.cd11 <> 0 AND iWCS.cd22 <> 0 THEN
        FUNCTION = 1
    END IF


END FUNCTION

FUNCTION WCSGetValue(dline AS STRING) AS DOUBLE
    LOCAL p AS LONG
    p = INSTR(dline, "=")
    IF p THEN
        FUNCTION = VAL(MID$(dline, p + 1))
    END IF
END FUNCTION


SUB StreakFastBackgroundMAD ( _
        BYVAL pImg AS LONG, _
        BYVAL iiWidth AS LONG, _
        BYVAL iiHeight AS LONG, _
        BYVAL istep AS LONG, _
        BYREF background AS DOUBLE, _
        BYREF sigma AS DOUBLE)

    LOCAL maxSamp AS LONG
    LOCAL icount AS LONG
    LOCAL x AS LONG, y AS LONG
    LOCAL buf() AS LONG
    LOCAL dev() AS LONG
    LOCAL v AS INTEGER
    LOCAL i AS LONG
    LOCAL addr1 AS LONG

    ' ---- Estimate maximum samples
    maxSamp = (iiWidth \ istep + 1) * (iiHeight \ istep + 1)
    REDIM buf(0 TO maxSamp - 1)

    icount = 0

    ' ---- Subsample image safely using byte addressing
    FOR y = 0 TO iiHeight - 1 STEP istep
        FOR x = 0 TO iiWidth - 1 STEP istep

            addr1 = pImg + ((y * iiWidth + x) * 2)

            v = PEEK(INTEGER, addr1)

            buf(icount) = v
            INCR icount

        NEXT
    NEXT

    IF icount < 10 THEN EXIT SUB

    REDIM PRESERVE buf(0 TO icount - 1)

    ' ---- Median background
    ARRAY SORT buf()
    background = buf(icount \ 2)

    ' ---- MAD computation
    REDIM dev(0 TO icount - 1)

    FOR i = 0 TO icount - 1
        dev(i) = ABS(buf(i) - background)
    NEXT

    ARRAY SORT dev()

    sigma = dev(icount \ 2) * 1.4826##

END SUB





FUNCTION FITSLoadInt16 (BYVAL fname AS STRING, _
                         img() AS INTEGER, _
                         BYREF iiWidth AS LONG, _
                         BYREF iiHeight AS LONG,_
                         BYREF iDateTime AS STRING,_
                         BYREF iExposure AS DOUBLE,_
                         BYREF iGain AS DOUBLE,_
                         BYREF SiteLat AS DOUBLE,_
                         BYREF SiteLong AS DOUBLE,_
                         BYREF SiteElev AS DOUBLE) AS LONG

    LOCAL ff AS LONG
    LOCAL card AS STRING * 80
    LOCAL bitpix AS LONG
    LOCAL naxis AS LONG
    LOCAL naxis1 AS LONG, naxis2 AS LONG
    LOCAL headerBytes AS LONG
    LOCAL imgbytes AS LONG
    LOCAL raw AS STRING
    LOCAL pSrc AS BYTE PTR
    LOCAL pDst AS INTEGER PTR
    LOCAL i AS LONG
    LOCAL temp AS INTEGER
    LOCAL b1 AS LONG, b2 AS LONG
    LOCAL v AS INTEGER
    LOCAL w, h, maxW, MaxH, maxVal AS LONG
    LOCAL hfile, posend, headersize AS LONG

    ON ERROR GOTO loadError


    IF LEN(DIR$(fname)) THEN


       ELSE

         GOTO loaderror
    END IF

    ff = FREEFILE
    OPEN fname FOR BINARY AS #ff
    IF LOF(ff)<2280 THEN ' bad file
        CLOSE # ff
        GOTO loaderror
    END IF

    ' -------- Parse FITS header --------
    DO
        GET$ #ff, 80, card

        IF LEFT$(card,6) = "BITPIX" THEN bitpix = VAL(MID$(card,11,22))
        IF LEFT$(card,6) = "NAXIS " THEN naxis  = VAL(MID$(card,11,22))
        IF LEFT$(card,6) = "NAXIS1" THEN naxis1 = VAL(MID$(card,11,22))
        IF LEFT$(card,6) = "NAXIS2" THEN naxis2 = VAL(MID$(card,11,22))
        IF LEFT$(card,8) = "DATE-OBS" THEN iDateTime = TRIM$(MID$(card,12,23))
        IF LEFT$(card,8) = "EXPOSURE" THEN iExposure = VAL(MID$(card,11,20))
        IF LEFT$(card,4) = "GAIN" THEN iGain = VAL(MID$(card,11,20))
        IF LEFT$(card,7) = "SITELAT" THEN SiteLat = VAL(MID$(card,11,20))
        IF LEFT$(card,7) = "SITELON" THEN SiteLong = VAL(MID$(card,11,20))
        IF LEFT$(card,7) = "SITEELE" THEN SiteElev = VAL(MID$(card,11,20))

        IF LEFT$(card,3) = "END" THEN EXIT DO

    LOOP


    ' header must end on 2880 boundary
    posEnd = SEEK(ff) - 1

    headerSize = ((posEnd + 2879) \ 2880) * 2880

      SEEK #ff, headerSize + 1
    IF bitpix <> 16 THEN
        CLOSE #ff
        FUNCTION = -2
        EXIT FUNCTION
    END IF

    iiWidth  = naxis1
    iiHeight = naxis2

    imgBytes = iiWidth * iiHeight * 2


    ' -------- Read Entire Image Block --------

     raw = SPACE$(imgBytes)
     GET$ #ff, imgBytes, raw

     CLOSE #ff


     ' -------- Allocate Output Array --------

      REDIM img(0 TO iiWidth-1, 0 TO iiHeight-1) AS INTEGER


     ' -------- Byte Swap In Memory --------
     pSrc = STRPTR(raw)

   FOR i = 0 TO iiWidth*iiHeight - 1

        b1 = PEEK(BYTE, pSrc)        ' high byte
        b2 = PEEK(BYTE, pSrc + 1)    ' low byte

        v = b1 * 256 + b2
        IF v > 32767 THEN v = v - 65536

        img(i MOD iiWidth, i \ iiWidth) = v

        pSrc = pSrc + 2

   NEXT

    maxval=0
    FOR w=0 TO iiWidth-1
        FOR h=0 TO iiHeight-1
            IF img(w,h)> Maxval THEN
               maxw= w
               maxh=h
               maxval=  img(w,h)
            END IF
        NEXT h
    NEXT w


    FUNCTION = -1
    EXIT FUNCTION
   LoadError:
    FUNCTION=0


END FUNCTION


SUB StreakLabelComponents( _
        mask() AS BYTE, _
        BYVAL iWidth AS LONG, _
        BYVAL iHeight AS LONG, _
        sImg() AS INTEGER, _
        BYVAL background AS DOUBLE, _
        BYVAL sigma AS DOUBLE, _
        BYVAL minArea AS LONG, _
        BYREF bestStartX AS DOUBLE, _
        BYREF bestStartY AS DOUBLE, _
        BYREF bestEndX AS DOUBLE, _
        BYREF bestEndY AS DOUBLE, _
        BYREF bestMidX AS DOUBLE, _
        BYREF bestMidY AS DOUBLE, _
        BYREF bestPeakSNR AS DOUBLE, _
        BYREF bestMeanSNR AS DOUBLE, _
        BYREF bestElongation AS DOUBLE, _
        BYREF bestLength AS DOUBLE)

    LOCAL labels() AS LONG
    REDIM labels(0 TO iWidth-1, 0 TO iHeight-1)

    LOCAL stack() AS LONG
    LOCAL pixelList() AS LONG

    REDIM stack(0 TO iWidth*iHeight-1)
    REDIM pixelList(0 TO iWidth*iHeight-1)

    LOCAL dx() AS LONG
    LOCAL dy() AS LONG

    REDIM dx(7)
    REDIM dy(7)

    dx(0)=-1: dy(0)=-1
    dx(1)= 0: dy(1)=-1
    dx(2)= 1: dy(2)=-1
    dx(3)=-1: dy(3)= 0
    dx(4)= 1: dy(4)= 0
    dx(5)=-1: dy(5)= 1
    dx(6)= 0: dy(6)= 1
    dx(7)= 1: dy(7)= 1

    LOCAL x,y,lbl AS LONG
    LOCAL sp, i AS LONG
    LOCAL xx, yy, nx, ny AS LONG

    bestElongation = 0
    lbl = 0

    FOR y = 0 TO iHeight-1
        FOR x = 0 TO iWidth-1

            IF mask(x,y)<>0 AND labels(x,y)=0 THEN

                INCR lbl

                ' ---- accumulators ----
                LOCAL sumX AS DOUBLE, sumY AS DOUBLE
                LOCAL sumXX AS DOUBLE, sumYY AS DOUBLE, sumXY AS DOUBLE
                LOCAL sumSignal AS DOUBLE, maxSignal AS DOUBLE
                LOCAL pixelCount AS LONG

                sumX=0: sumY=0
                sumXX=0: sumYY=0: sumXY=0
                sumSignal=0: maxSignal=0
                pixelCount=0

                ' ---- flood-fill work stack ----
                sp = 0
                labels(x,y)=lbl
                stack(sp)=y*iWidth + x
                INCR sp

                DO WHILE sp>0

                    DECR sp
                    yy = stack(sp) \ iWidth
                    xx = stack(sp) MOD iWidth

                    ' ---- store pixel separately ----
                    pixelList(pixelCount) = yy*iWidth + xx
                    INCR pixelCount

                    ' ---- accumulate stats ----
                    sumX = sumX + xx
                    sumY = sumY + yy
                    sumXX = sumXX + xx*xx
                    sumYY = sumYY + yy*yy
                    sumXY = sumXY + xx*yy

                    LOCAL signal AS DOUBLE
                    signal = sImg(xx,yy) - background
                    IF signal < 0 THEN signal=0

                    sumSignal = sumSignal + signal
                    IF signal > maxSignal THEN maxSignal = signal

                    ' ---- neighbors ----
                    FOR i=0 TO 7
                        nx = xx + dx(i)
                        ny = yy + dy(i)

                        IF nx>=0 AND nx<iWidth AND ny>=0 AND ny<iHeight THEN
                            IF mask(nx,ny)<>0 AND labels(nx,ny)=0 THEN
                                labels(nx,ny)=lbl
                                stack(sp)=ny*iWidth + nx
                                INCR sp
                            END IF
                        END IF
                    NEXT

                LOOP

                IF pixelCount < minArea THEN ITERATE FOR

                ' ---- centroid ----
                LOCAL cx AS DOUBLE, cy AS DOUBLE
                cx = sumX / pixelCount
                cy = sumY / pixelCount

                ' ---- covariance ----
                LOCAL sxx AS DOUBLE, syy AS DOUBLE, sxy AS DOUBLE
                sxx = (sumXX/pixelCount) - cx*cx
                syy = (sumYY/pixelCount) - cy*cy
                sxy = (sumXY/pixelCount) - cx*cy

                LOCAL itrace AS DOUBLE, diff AS DOUBLE, sqrtTerm AS DOUBLE
                LOCAL lambda1 AS DOUBLE, lambda2 AS DOUBLE

                itrace = sxx + syy
                diff = sxx - syy
                sqrtTerm = SQR(diff*diff + 4*sxy*sxy)

                lambda1 = 0.5*(itrace + sqrtTerm)
                lambda2 = 0.5*(itrace - sqrtTerm)

                IF lambda1<0 THEN lambda1=0
                IF lambda2<0 THEN lambda2=0

                LOCAL majorLength AS DOUBLE
                LOCAL minorLength AS DOUBLE

                majorLength = SQR(lambda1)
                minorLength = SQR(lambda2)

                IF minorLength<=0 THEN minorLength=0.0001##

                LOCAL elongation AS DOUBLE
                elongation = majorLength/minorLength

                LOCAL meanSNR AS DOUBLE
                LOCAL peakSNR AS DOUBLE

                meanSNR = (sumSignal/pixelCount)/sigma
                peakSNR = maxSignal/sigma

                ' ---- projection for endpoints ----
                LOCAL majorAngle AS DOUBLE
                majorAngle = 0.5 * ATAN2(2*sxy, sxx - syy)

                LOCAL vx AS DOUBLE, vy AS DOUBLE
                vx = COS(majorAngle)
                vy = SIN(majorAngle)

                LOCAL projMin AS DOUBLE, projMax AS DOUBLE
                LOCAL firstProj AS LONG
                LOCAL startX AS DOUBLE, startY AS DOUBLE
                LOCAL endX AS DOUBLE, endY AS DOUBLE

                firstProj = 1

                FOR i=0 TO pixelCount-1

                    yy = pixelList(i) \ iWidth
                    xx = pixelList(i) MOD iWidth

                    LOCAL proj AS DOUBLE
                    proj = (xx-cx)*vx + (yy-cy)*vy

                    IF firstProj THEN
                        projMin=proj: projMax=proj
                        startX=xx: startY=yy
                        endX=xx: endY=yy
                        firstProj=0
                    ELSE
                        IF proj<projMin THEN
                            projMin=proj
                            startX=xx: startY=yy
                        END IF
                        IF proj>projMax THEN
                            projMax=proj
                            endX=xx: endY=yy
                        END IF
                    END IF

                NEXT

                LOCAL ObservedLength AS DOUBLE
                ObservedLength = projMax - projMin

                ' ---- choose best ----
                IF elongation>bestElongation AND peakSNR>5 THEN

                    bestElongation = elongation
                    bestStartX = startX
                    bestStartY = startY
                    bestEndX = endX
                    bestEndY = endY
                    bestMidX = cx
                    bestMidY = cy
                    bestPeakSNR = peakSNR
                    bestMeanSNR = meanSNR
                    bestLength = ObservedLength

                END IF

            END IF

        NEXT
    NEXT

END SUB







'------------------------------------------------------------
' PCA_Component
' Computes centroid, major axis angle, and length scales
' for a single labeled component in a 2D label array.
'------------------------------------------------------------
SUB StreakPCAComponent( _
        BYVAL targetLabel AS LONG, _
        labels() AS LONG, _
        BYVAL iWidth AS LONG, BYVAL iHeight AS LONG, _
        BYREF cx AS DOUBLE, BYREF cy AS DOUBLE, _
        BYREF majorAngle AS DOUBLE, _
        BYREF majorLength AS DOUBLE, BYREF minorLength AS DOUBLE)

    LOCAL x AS LONG, y AS LONG
    LOCAL sumX AS DOUBLE, sumY AS DOUBLE
    LOCAL iCount AS LONG

    ' --- Step 1: compute centroid ---
    sumX = 0
    sumY = 0
    iCount = 0

    FOR y = 0 TO iHeight-1
        FOR x = 0 TO iWidth-1
            IF labels(x,y) = targetLabel THEN
                sumX = sumX + x
                sumY = sumY + y
                INCR iCount
            END IF
        NEXT
    NEXT

    IF iCount = 0 THEN EXIT SUB

    cx = sumX / iCount
    cy = sumY / iCount

    ' --- Step 2: compute covariance matrix ---
    LOCAL dx AS DOUBLE, dy AS DOUBLE
    LOCAL sxx AS DOUBLE, syy AS DOUBLE, sxy AS DOUBLE
    sxx = 0: syy = 0: sxy = 0

    FOR y = 0 TO iHeight-1
        FOR x = 0 TO iWidth-1
            IF labels(x,y) = targetLabel THEN
                dx = x - cx
                dy = y - cy
                sxx = sxx + dx*dx
                syy = syy + dy*dy
                sxy = sxy + dx*dy
            END IF
        NEXT
    NEXT

    sxx = sxx / iCount
    syy = syy / iCount
    sxy = sxy / iCount

    ' --- Step 3: compute eigenvalues ---
    LOCAL itrace AS DOUBLE, diff AS DOUBLE, sqrtTerm AS DOUBLE
    LOCAL lambda1 AS DOUBLE, lambda2 AS DOUBLE
    itrace = sxx + syy
    diff = sxx - syy
    sqrtTerm = SQR(diff*diff + 4*sxy*sxy)

    lambda1 = 0.5 * (itrace + sqrtTerm)  ' major axis
    lambda2 = 0.5 * (itrace - sqrtTerm)  ' minor axis

    majorLength = SQR(lambda1)
    minorLength = SQR(lambda2)

    ' --- Step 4: major axis angle ---
    ' atan2 returns angle in radians
    majorAngle = 0.5 * ATaN2(2*sxy, sxx - syy)

END SUB


SUB StreakEvaluateQuality( _
        labels() AS LONG, _
        BYVAL targetLabel AS LONG, _
        BYVAL iWidth AS LONG, _
        BYVAL iHeight AS LONG, _
        sImg() AS INTEGER, _
        BYVAL background AS DOUBLE, _
        BYVAL sigma AS DOUBLE, _
        BYREF peakSNR AS DOUBLE, _
        BYREF meanSNR AS DOUBLE, _
        BYREF elongation AS DOUBLE, _
        BYREF ObservedLength AS DOUBLE)


    LOCAL x AS LONG, y AS LONG
    LOCAL cx AS DOUBLE, cy AS DOUBLE
    LOCAL majorAngle AS DOUBLE, majorLength AS DOUBLE, minorLength AS DOUBLE
    LOCAL vx AS DOUBLE, vy AS DOUBLE
    LOCAL proj AS DOUBLE
    LOCAL projMin AS DOUBLE, projMax AS DOUBLE
    LOCAL first AS LONG
    LOCAL signal AS DOUBLE
    LOCAL sumSignal AS DOUBLE
    LOCAL maxSignal AS DOUBLE
    LOCAL icount AS LONG

    ' --- PCA ---
    StreakPCAComponent targetLabel, labels(), iWidth, iHeight, _
                  cx, cy, majorAngle, majorLength, minorLength

    IF minorLength <= 0 THEN minorLength = 0.0001##

    elongation = majorLength / minorLength

    vx = COS(majorAngle)
    vy = SIN(majorAngle)

    first = 1
    sumSignal = 0##
    maxSignal = 0##
    icount = 0

    FOR y = 0 TO iHeight - 1
        FOR x = 0 TO iWidth - 1

            IF labels(x,y) = targetLabel THEN

                signal = sImg(x,y) - background
                IF signal < 0 THEN signal = 0

                sumSignal = sumSignal + signal
                IF signal > maxSignal THEN maxSignal = signal
                INCR icount

                proj = (x - cx)*vx + (y - cy)*vy

                IF first THEN
                    projMin = proj
                    projMax = proj
                    first = 0
                ELSE
                    IF proj < projMin THEN projMin = proj
                    IF proj > projMax THEN projMax = proj
                END IF

            END IF

        NEXT
    NEXT

       ' MSGBOX "icount" + STR$(icount)+"  sigma="+str$(sigma)
    IF icount = 0 OR sigma <= 0 THEN EXIT SUB

    meanSNR = (sumSignal / icount) / sigma
    peakSNR = maxSignal / sigma

    ObservedLength = projMax - projMin

END SUB


FUNCTION StreakOnEdge(siWidth AS LONG, siHeight AS LONG, StartX AS DOUBLE, StartY AS DOUBLE, EndX AS DOUBLE, EndY AS DOUBLE ) AS LONG
    LOCAL eMARGIN AS LONG
    LOCAL edgeFlag AS LONG

    eMARGIN = gStreakMinMargin
    edgeFlag = 0

    IF startX < emargin THEN edgeFlag = 1
    IF startY < emargin THEN edgeFlag = 1
    IF endX > siWidth - emargin THEN edgeFlag = 1
    IF endY > siHeight - emargin THEN edgeFlag = 1

    FUNCTION=edgeFlag

END FUNCTION

SUB StreakPixelToRADec (x AS DOUBLE, y AS DOUBLE, w AS WCS, ra AS DOUBLE, dec AS DOUBLE)

    LOCAL dx AS DOUBLE
    LOCAL dy AS DOUBLE

    dx = x - w.crpix1
    dy = y - w.crpix2

    ra  = w.crval1 + (w.cd11 * dx + w.cd12 * dy)
    dec = w.crval2 + (w.cd21 * dx + w.cd22 * dy)

END SUB


SUB StreakCameraApply
    LOCAL temp AS STRING

    CONTROL GET TEXT hLeapFrogTab, %txtStreakExposure TO temp
    gStreakExposure=VAL(temp)
    CONTROL GET TEXT hLeapFrogTab, %txtStreakGain TO temp
    gStreakGain=VAL(temp)


END SUB

SUB StreakQualityApply
    LOCAL temp AS STRING

    CONTROL GET TEXT hStreakAnalysisSetupTab, %txtEdgeMargin TO temp
    gStreakMinMargin=VAL(temp)
    CONTROL GET TEXT hStreakAnalysisSetupTab, %txtStreakMinLength TO temp
    gStreakMinLength=VAL(temp)
    CONTROL GET TEXT hStreakAnalysisSetupTab, %txtStreakMinPeakSNR TO temp
    gStreakMinPeakSNR=VAL(temp)
     CONTROL GET TEXT hStreakAnalysisSetupTab, %txtStreakMinElongation TO temp
    gStreakMinElongation=VAL(temp)

END SUB


SUB StreakExpectedPos(NORAD AS LONG, iDateTime AS STRING, iExposure AS DOUBLE, ExpectedRA AS DOUBLE, ExpectedDec AS DOUBLE, _
                      ExpectedRange AS DOUBLE, ExpectedLength AS DOUBLE)
    LOCAL MidJD, startJD, EndJD AS DOUBLE, t AS LONG, SatelliteIndex AS LONG
    LOCAL Usergeodetic, azmel, pPos, pvel, obsset AS vector
    LOCAL startRA,StartDec,EndRA, EndDec, Sep, PixelSize, FocalLength,ArcSecPerPixel AS DOUBLE, temp AS STRING


    startJD=StreakJD(iDateTime)
    MidJD=StartJD+(iExposure/24/60/60/2)
    EndJD=StartJD+(iExposure/24/60/60)

    CONTROL GET TEXT hPlateSolveSetup,%txtScopeFL TO temp
    FocalLength= VAL(temp)

    CONTROL GET TEXT hPlateSolveSetup,%txtCameraPixel TO temp
    PixelSize = VAL(temp)



    FOR t=1 TO SatelliteCount
        IF Norad=VAL(alltrim(Norad_pc(t))) THEN
            SatelliteIndex=t
            EXIT FOR
        END IF
    NEXT t

   iflag_PC(t) = 1     'was (arg)
    'Get Site Location
    Usergeodetic.v1 =site(CurrentSite).sLatitude
    Usergeodetic.v2 =site(CurrentSite).sLongitude
    Usergeodetic.v3 = site(CurrentSite).sElevation
    'Latitude= site(CurrentSite).sLatitude
    'Longitude= site(CurrentSite).sLongitude

    Usergeodetic.v1 = Radians(Usergeodetic.v1)
    Usergeodetic.v2 = Radians(Modulus(Usergeodetic.v2,360))
    Usergeodetic.v3 = Usergeodetic.v3*0.001

    ConvertSatelliteData SatelliteIndex,24
    SGP Midjd,pPos,pVel,24
    ConvertSatelliteState pPos,pVel
    CalculateRADec ppos ,pvel ,Usergeodetic ,Midjd,obsset,24
     calculateObs ppos ,pvel ,Usergeodetic ,Midjd,azmel,24

    ExpectedRA=degrees(obsset.v1)
    ExpectedDec=degrees(obsset.v2)
    ExpectedRange=azmel.v3

       'Change to J2000
    ExpectedRa=ExpectedRA/15
    TopoJ2000 (ExpectedRA,ExpectedDec)
    ExpectedRA=ExpectedRA*15

    IF focalLength<>0 AND Pixelsize <>0 THEN
        ArcSecPerPixel=(PixelSize/FocalLength)*206.265
        'Get Start Pos
        ConvertSatelliteData SatelliteIndex,24
        SGP Startjd,pPos,pVel,24
        ConvertSatelliteState pPos,pVel
        CalculateRADec ppos ,pvel ,Usergeodetic ,Midjd,obsset,24
        StartRA=degrees(obsset.v1)
        StartDec=degrees(obsset.v2)

        'Get End Pos
        ConvertSatelliteData SatelliteIndex,24
        SGP Endjd,pPos,pVel,24
        ConvertSatelliteState pPos,pVel
        CalculateRADec ppos ,pvel ,Usergeodetic ,Midjd,obsset,24
        EndRA=degrees(obsset.v1)
        EndDec=degrees(obsset.v2)
        ExpectedLength =StreakLengthPixels (StartRA, StartDec, EndRA, EndDec, ArcSecPerPixel)
    ELSE
        ExpectedLength=-1
    END IF

END SUB



FUNCTION StreakGood(lstllrow AS LONG) AS LONG
    LOCAL hLV AS DWORD

    CONTROL HANDLE hOrbitCreateTab, %lstStreakDetermineOrbit TO hLV
    FUNCTION = ListView_GetCheckState(hLV, lstllrow)

END FUNCTION

FUNCTION StreakAltitude(ObjectRa AS DOUBLE, ObjectDec AS DOUBLE, jd AS DOUBLE, lat AS DOUBLE, longitude AS DOUBLE) AS STRING
     LOCAL r AS DOUBLE, d AS DOUBLE, ha AS DOUBLE,phi AS DOUBLE, sina AS DOUBLE, x AS DOUBLE, y AS DOUBLE
     LOCAL s1 AS DOUBLE, c1 AS DOUBLE, c2 AS DOUBLE, a9 AS DOUBLE, p AS DOUBLE, b AS DOUBLE, t5 AS DOUBLE, t AS DOUBLE
     LOCAL h AS DOUBLE,s2 AS DOUBLE, LocalLatitude AS DOUBLE, lsidereal AS DOUBLE, MeanSidereal, ObjectAlt AS DOUBLE

    t=(jd-2451545)/36525
    MeanSideReal= 280.46061837+360.98564736629*(jd-2451545)+0.000387933*T^2-T^3/38710000
    meanSideReal=MeanSideReal+ Longitude

    WHILE MeanSideReal<0
        MeanSideReal=MeanSideReal+360
    WEND
    WHILE MeanSideReal=>360
        MeanSideReal=MeanSideReal-360
    WEND

    lSideReal=d2r(MeanSideReal)
     LocalLatitude= lat

     phi=radians(LocalLatitude)
     r = radians(ObjectRA)
     d = radians(ObjectDec)
     ha= lSidereal-r
     b=phi
     t5=ha
     p=3.14159265

      S1=SIN(B)*SIN(D)
      S1=S1+COS(B)*COS(D)*COS(T5)
      C1=1-S1*S1
      IF C1>0 THEN C1=SQR(C1)
      IF C1<=0 THEN
         H=SGN(S1)*P/2
      END IF
      H=ATN(S1/C1)


       ObjectAlt=r2d(h)
       IF ObjectRa+ObjectDec=0 THEN
           FUNCTION="   ------"
       ELSE
           FUNCTION = FORMAT$(ObjectAlt,"#.0")
       END IF
END FUNCTION


SUB LoadStreakListView(fileName AS STRING)

    LOCAL hLV AS DWORD
    LOCAL hFile,tt AS LONG
    LOCAL LineTxt, temp, Altitude AS STRING
    LOCAL f() AS STRING
    LOCAL i AS LONG
    LOCAL p AS LONG
    LOCAL startPos AS LONG
    LOCAL rejectFlag AS LONG
    LOCAL lrow AS LONG
    LOCAL ExposureJD AS DOUBLE
    LOCAL dRA, dDec, mRA, mDec, mag, uRA, uDec, Atrk, Xtrk, totalres AS DOUBLE

    StreakOrbitalClear 1

    CONTROL HANDLE hOrbitCreateTab,%lstStreakDetermineOrbit TO hLV
    LISTVIEW RESET hOrbitCreateTab,%lstStreakDetermineOrbit

    IF DIR$(fileName) = "" THEN EXIT SUB

    gStreakCount = 0
    gLoadingStreakList=1
    DIM StreakObs(1 TO 1)

    hFile = FREEFILE
    OPEN fileName FOR INPUT AS #hFile


    LINE INPUT #hFile, LineTxt  'version header
    LINE INPUT #hFile, LineTxt  'version
    LINE INPUT #hFile, LineTxt  'Site Header
    LINE INPUT #hFile, LineTxt  ' Header
    LINE INPUT #hFile, LineTxt  'Site

    ' Parse tab-delimited fields
    DIM f(1 TO 3)
    FOR i = 1 TO 3
        f(i) = ""
    NEXT

    startPos = 1

    FOR i = 1 TO 3
        p = INSTR(startPos, LineTxt, CHR$(9))

        IF p = 0 THEN
            f(i) = MID$(LineTxt, startPos)
            EXIT FOR
        ELSE
            f(i) = MID$(LineTxt, startPos, p - startPos)
            startPos = p + 1
        END IF
    NEXT

    ' Assign values
    gStreakLatitude  = VAL(f(1))
    gStreakLongitude = VAL(f(2))
    gStreakElevation = VAL(f(3))

    REDIM f(1 TO 27) '1 to 26
    LINE INPUT #hFile, LineTxt  'TLE Header
    LINE INPUT #hFile, LineTxt  'TLE Header
    LINE INPUT #hFile, LineTxt  'TLE


    LOCAL t() AS STRING
    DIM t(1 TO 18)                    '(1 TO 15)

    startPos = 1
    FOR i = 1 TO 18
        p = INSTR(startPos, LineTxt, CHR$(9))
        IF p = 0 THEN
            t(i) = MID$(LineTxt, startPos)
            EXIT FOR
        ELSE
            t(i) = MID$(LineTxt, startPos, p - startPos)
            startPos = p + 1
        END IF
    NEXT

    '----------------------------------
    ' Extract TLE values
    '----------------------------------
    LOCAL NORAD AS STRING
    LOCAL EpochYear AS STRING, EpochDay2 AS STRING
    LOCAL Incl AS DOUBLE, RAAN AS DOUBLE, ECC AS STRING
    LOCAL ARGP AS DOUBLE, MA AS DOUBLE, MM AS DOUBLE
    LOCAL BSTAR AS STRING, MM1 AS DOUBLE, MM2 AS STRING
    LOCAL EphemType AS STRING, Elset AS STRING, Rev AS STRING

    NORAD = t(1)
    temp  = t(1)   ' keep your existing NORAD usage

    EpochYear = RIGHT$("0" + t(2), 2)
    EpochDay2  = t(3)


    Incl = VAL(t(4))
    RAAN = VAL(t(5))
    REPLACE "." WITH "" IN t(6)

    ECC  = RIGHT$("0000000" +t(6), 7)
    ARGP = VAL(t(7))
    MA   = VAL(t(8))
    MM   = VAL(t(9))

    BSTAR = (t(10))
    MM1   = VAL(t(11))
    MM2   = t(12)

    EphemType = t(13)
    Elset     = RIGHT$("0000" + t(14), 4)
    Rev       = RIGHT$("00000" + t(15), 5)

    CONTROL SET TEXT hOrbitCreateTab, %streakFrmOrbit, "Orbital Elements for NORAD: "+TRIM$(temp)

    'Load Orignal TLE
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtEpochOrig, EpochYear+" "+t(3)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtInclOrig, FORMAT$(VAL(t(4)),"000.0000")
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtRAANOrig, t(5)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtEccOrig, ECC
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtArgPerOrig, t(7)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtMeanAnomOrig, t(8)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtMeanMotionOrig, t(9)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtRevNumOrig,rev
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtMM1Orig,t(11)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtMM2Orig, t(12)
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtBSTAROrig, t(10)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtEphemerisOrig, t(13)
    CONTROL SET TEXT hOrbitCreateTab, %streakTxtElsetOrig, Elset

    '----------------------------------
    ' Build proper TLE
    '----------------------------------

    StreakSatelliteData(1)=SPACE$(44)
    StreakSatelliteData(2)="1"+SPACE$(68)
    StreakSatelliteData(3)="2"+SPACE$(68)

    StreakSatelliteData(1) = LEFT$(t(16)+SPACE$(24),24) +LEFT$(Norad+SPACE$(9),9)+ LEFT$(t(17)+SPACE$(10),10)

    'take century off objectid
   ' t(17)=RIGHT$(t(17),LEN(t(17))-2)
   ' t(17)=REMOVE$(t(17),"-")
    LOCAL line1 AS STRING
    line1 = "1 " +_
        LEFT$(NORAD+SPACE$(5),5)+_
        LEFT$(t(18)+" ",1)+" "+ _
        LEFT$(t(17)+SPACE$(8),8)+" "+ _
        RIGHT$("00"+EpochYear,2) + _
        FORMAT$(VAL(EpochDay2), "000.00000000") + " " + _
        FORMAT$(MM1, "+.00000000;-.00000000") + " " + _
        RIGHT$(SPACE$(8)+MM2,8) + " " + _
        RIGHT$(SPACE$(8)+BSTAR,8) + " " + _
        LEFT$(EphemType+" ",1) + " " + _
        Elset

    line1 = LEFT$(line1 + SPACE$(68), 68)
    StreakSatelliteData(2) = line1 + TLEChecksum(line1)

    LOCAL line2 AS STRING
    line2 = "2 " + _
    LEFT$(NORAD+SPACE$(5),5) + " " + _
    FORMAT$(Incl, "000.0000") + " " + _
    FORMAT$(RAAN, "000.0000") + " " + _
    ECC + " " + _
    FORMAT$(ARGP, "000.0000") + " " + _
    FORMAT$(MA, "000.0000") + " " + _
    FORMAT$(MM, "00.00000000") + _
    Rev

    line2 = LEFT$(line2 + SPACE$(68), 68)
    StreakSatelliteData(3) = line2 + TLEChecksum(line2)

    ConvertStreaksDataPC  'Loads into Arg 0

    writeaudit 1,"Streak TLE Data"

    writeaudit 1,StreakSatelliteData(1)
    writeaudit 1,StreakSatelliteData(2)
    writeaudit 1,StreakSatelliteData(3)


    LINE INPUT #hFile, LineTxt  'Observations header
    LINE INPUT #hFile, LineTxt  'Observations h

  ' WriteAudit 1, "At Loadg_dX=" + STR$(g_dX) + " g_dY=" + STR$(g_dY)

    DO UNTIL EOF(hFile)

        LINE INPUT #hFile, LineTxt


        '------------------------
        ' Parse tab fields
        '------------------------
        FOR i = 1 TO 27
            f(i) = ""
        NEXT

        startPos = 1

        FOR i = 1 TO 27

            p = INSTR(startPos,LineTxt,CHR$(9))

            IF p = 0 THEN
                f(i) = MID$(LineTxt,startPos)
                EXIT FOR
            ELSE
                f(i) = MID$(LineTxt,startPos,p-startPos)
                startPos = p + 1
            END IF

        NEXT

        '------------------------
        ' Glrow observation array
        '------------------------
        INCR gStreakCount
        REDIM PRESERVE StreakObs(1 TO gStreakCount)

        '------------------------
        ' Fill observation record
        '------------------------
      '  WITH StreakObs(gStreakCount)

            StreakObs(gStreakCount).Norad      = TRIM$(temp)
            StreakObs(gStreakCount).sImage      = f(1)
          '  ExposureJD= streakjd(f(3)) +  (VAL(f(24)) / 2) / 86400##
            StreakObs(gStreakCount).MidTimeJD = VAL(f(3))
            'StreakObs(gStreakCount).MidTimeJD = StreakObs(gStreakCount).MidTimeJD + (0.3/86400.0##)
            StreakObs(gStreakCount).StartTimeJD = StreakObs(gStreakCount).MidTimeJD - ((VAL(f(24)) / 2.0##) / 86400.0##)
            StreakObs(gStreakCount).EndTimeJD   = StreakObs(gStreakCount).MidTimeJD + ((VAL(f(24)) / 2.0##) / 86400.0##)



            StreakObs(gStreakCount).PeakSNR    = VAL(f(10))
            StreakObs(gStreakCount).MeanSNR    = VAL(f(11))
            StreakObs(gStreakCount).Elongation = VAL(f(12))
            StreakObs(gStreakCount).Length     = VAL(f(13))

            IF VAL(f(15))=0 THEN
                StreakObs(gStreakCount).UseObs     = 1
            ELSE
                StreakObs(gStreakCount).UseObs     = 0
            END IF

            StreakObs(gStreakCount).MidRA      = VAL(f(16))
            StreakObs(gStreakCount).MidDec     = VAL(f(17))

            StreakObs(gStreakCount).StartRA      = VAL(f(18))    '
            StreakObs(gStreakCount).StartDec     = VAL(f(19))

            StreakObs(gStreakCount).EndRA      = VAL(f(20))      '
            StreakObs(gStreakCount).EndDec     = VAL(f(21))

            StreakObs(gStreakCount).ExpRA      = VAL(f(22))
            StreakObs(gStreakCount).ExpDec     = VAL(f(23))
            StreakObs(gStreakCount).ExpRange   = VAL(f(24))

            StreakObs(gStreakCount).Exposure   = VAL(f(25))
            StreakObs(gStreakCount).Gain       = VAL(f(26))


      CALL ComputeOneResidual(1,gStreakCount, Atrk, Xtrk,1)
        TotalRes = SQR(Atrk * Atrk + Xtrk * Xtrk)
        Altitude=StreakAltitude(StreakObs(gStreakCount).MidRA, StreakObs(gStreakCount).MidDec ,StreakObs(gStreakCount).MidTimeJD _
                               , gStreakLatitude, gStreakLongitude  )

        lrow = gStreakCount

        LISTVIEW INSERT ITEM hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,0,""

        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,2, CalendarDate(VAL(f(3)))+"  "+TimeOfDay(VAL(f(3)), 1,4)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,3,f(10)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,4,f(11)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,5,f(12)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,6,f(13)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,7,f(14)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,8,f(25)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,9,f(26)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,10,Altitude
        'LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,10,f(16)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,11,FORMAT$(xtrk,"#.0")
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,12,FORMAT$(atrk,"#.0")
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,13,FORMAT$(TotalRes,"#.0") ' total res
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,lrow,14,f(27)

        ' checkbox
        ListView_SetCheckState hLV,lrow-1,StreakObs(lrow).UseObs

    LOOP

    gLoadingStreakList=0
    CLOSE #hFile

    CONTROL DISABLE hOrbitCreateTab,%bnXATrkOrig
    CONTROL DISABLE hOrbitCreateTab,%bnXATrkFit
    CONTROL DISABLE hOrbitCreateTab,%bnstreakSaveOrbital




END SUB



SUB AXTrkRefresh(Fitted AS LONG)
    LOCAL t AS LONG
    LOCAL ATrk, XTrk, TotalRes AS DOUBLE

    FOR t=1 TO  gStreakCount
        CALL ComputeOneResidual(1,t, Atrk, Xtrk, Fitted)
        TotalRes = SQR(Atrk * Atrk + Xtrk * Xtrk)
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,t,11,FORMAT$(xtrk,"#.0")
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,t,12,FORMAT$(atrk,"#.0")
        LISTVIEW SET TEXT hOrbitCreateTab,%lstStreakDetermineOrbit,t,13,FORMAT$(TotalRes,"#.0")
    NEXT t

END SUB



FUNCTION StreakFileGet AS LONG
    LOCAL defaultFile AS STRING, HomePath AS STRING, newFile AS STRING

    HomePath=gImagesPath
    DefaultFile=""
    DISPLAY OPENFILE hmaindlg, , , "Streak Data File", HomePath,CHR$("Files (*.strk)",0, "*.strk" ,0), _
       defaultfile, ".strk", %OFN_FILEMUSTEXIST OR %OFN_PATHMUSTEXIST  TO NewFile

    IF NewFile<>"" THEN
        LoadStreakListView NewFile
        CONTROL SET TEXT hOrbitCreateTab,%txtStreakFile, NewFile
        '
    END IF

END FUNCTION

'------------------------------------------------------------
' Compute pixel length of a satellite streak
'
' Inputs:
'   RA1, Dec1  = start position  (degrees)
'   RA2, Dec2  = end position    (degrees)
'   ArcSecPerPixel = plate scale (arcsec/pixel)
'
' Returns:
'   Pixel length of the streak (DOUBLE)
'------------------------------------------------------------

FUNCTION StreakLengthPixels ( _
    BYVAL RA1 AS DOUBLE, BYVAL Dec1 AS DOUBLE, _
    BYVAL RA2 AS DOUBLE, BYVAL Dec2 AS DOUBLE, _
    BYVAL ArcSecPerPixel AS DOUBLE) AS DOUBLE

    LOCAL dRA_deg     AS DOUBLE
    LOCAL dDec_deg    AS DOUBLE
    LOCAL DecAvg_rad  AS DOUBLE
    LOCAL dRA_arcsec  AS DOUBLE
    LOCAL dDec_arcsec AS DOUBLE
    LOCAL Sep_arcsec  AS DOUBLE

    '--- RA difference (degrees)
    dRA_deg = RA2 - RA1

    '--- Wrap-around correction (shortest path)
    IF dRA_deg > 180## THEN dRA_deg = dRA_deg - 360##
    IF dRA_deg < -180## THEN dRA_deg = dRA_deg + 360##

    '--- Dec difference (degrees)
    dDec_deg = Dec2 - Dec1

    '--- Average declination (radians)
    DecAvg_rad = ((Dec1 + Dec2) / 2##) * gDEG2RAD

    '--- Convert to arcseconds
    dRA_arcsec  = dRA_deg  * 3600##
    dDec_arcsec = dDec_deg * 3600##

    '--- Apply cosine(dec) correction to RA
    dRA_arcsec = dRA_arcsec * COS(DecAvg_rad)

    '--- Angular separation (arcseconds)
    Sep_arcsec = SQR(dRA_arcsec * dRA_arcsec + dDec_arcsec * dDec_arcsec)

    '--- Convert to pixels
    FUNCTION = Sep_arcsec / ArcSecPerPixel

END FUNCTION


'JD is exposure start time
SUB StreakDirection(JD AS STRING, NORAD AS LONG, BYREF StartRA AS DOUBLE, _
                           BYREF StartDec AS DOUBLE, BYREF EndRa AS DOUBLE, BYREF EndDec AS DOUBLE)
    LOCAL ExpectedRA, ExpectedDec, ExpectedRange,ExpectedLength, distance1, distance2, tempRA, tempDec AS DOUBLE

    StreakExpectedPos NORAD, JD, 0, ExpectedRA, ExpectedDec, ExpectedRange, ExpectedLength
    Distance1=SeparationArcMin(StartRA ,StartDec ,ExpectedRA,ExpectedDec)
    Distance2=SeparationArcMin(EndRA ,EndDec ,ExpectedRA,ExpectedDec)

    IF Distance1 > Distance2 THEN
        tempRa=StartRa
        tempDec=StartDec
        StartRa=EndRa
        StartDec=EndDec
        EndRa=tempRa
        EndDec=tempDec
        writeaudit 1, "Swapped streak enpoints"
    END IF

END SUB

FUNCTION ComputeTotalError() AS DOUBLE

    LOCAL i, StreakCnt AS LONG
    LOCAL RA_pred AS DOUBLE, Dec_pred AS DOUBLE, Range_pred AS DOUBLE
    LOCAL dRA AS DOUBLE, dDec AS DOUBLE
    LOCAL mRA AS DOUBLE, mDec AS DOUBLE, mag AS DOUBLE
    LOCAL uRA AS DOUBLE, uDec AS DOUBLE
    LOCAL Atrk AS DOUBLE, Xtrk AS DOUBLE
    LOCAL TotalError AS DOUBLE

    TotalError = 0
    StreakCnt=0

    FOR i = 1 TO gStreakCount

        IF StreakObs(i).UseObs > 0 THEN
            CALL ComputeOneResidual(1,i, Atrk, Xtrk,1)
            TotalError = TotalError + (Atrk*Atrk + Xtrk*Xtrk)
            INCR StreakCnt
        END IF
    NEXT

    WriteAudit 1, "Obs " + STR$(i) + _
     " Atrk=" + STR$(Atrk) + _
      " Xtrk=" + STR$(Xtrk)
    IF StreakCnt > 0 THEN

      FUNCTION = SQR(TotalError / StreakCnt)
    ELSE
      FUNCTION = 0

    END IF

END FUNCTION


FUNCTION ConvertStreaksDataPC AS LONG
    LOCAL iexp AS LONG,ibexp, isXML AS LONG
    LOCAL a1 AS DOUBLE,ao AS DOUBLE,del1 AS DOUBLE,delo AS DOUBLE,xnodp AS DOUBLE,temp AS DOUBLE
    LOCAL  abuf() AS STRING, arg AS LONG
    LOCAL dummy AS DOUBLE

   ' ClearSatelliteGlobalVariablesPC


    arg=0
    REDIM abuf(1 TO 2)

    abuf(1)= StreakSatelliteData(2)
    abuf(2)= StreakSatelliteData(3)

    IF LEN(StreakSatelliteData(1))>24 THEN
        isXML=1
        Norad_PC(arg) =TRIM$(MID$(StreakSatelliteData(1),25,9))
    ELSE
        isXML=0
        Norad_PC(arg) = MID$(abuf(1),3,5)     'this breaks when NORAD >5 long
    END IF

    SatelliteName_PC(arg)=MID$(StreakSatelliteData(1),1,24)
    IF isXML=0 THEN
        LaunchYear_PC(arg) = MID$(abuf(1),10,2)
        LaunchNo_PC(arg)=MID$(abuf(1),12,3)
        IF VAL(LaunchYear_PC(arg))>57 THEN
            LaunchYear_PC(arg)="19"+LaunchYear_PC(arg)
        ELSE
            LaunchYear_PC(arg)="20"+LaunchYear_PC(arg)
        END IF
        LaunchYear_PC(arg)=LaunchYear_PC(arg)+"-"+LaunchNo_PC(arg)+ TRIM$(MID$(StreakSatelliteData(2),15,3) )
    ELSE
       LaunchYear_PC(arg)=TRIM$(MID$(StreakSatelliteData(1),34,11))
    END IF

    IF isXML=0 THEN
        catnr_PC(arg)       = MID$(abuf(1),3,5)
    ELSE
        catnr_PC(arg)       =  Norad_PC(arg)
    END IF

    epoch_PC(arg)       = RealValue(abuf(1),19,14)

    julianepoch_PC(arg) = JulianDateofEpoch(epoch_PC(arg))

    xndt2o_PC(arg)   = RealValue(abuf(1),34,10)
    xndd6o_PC(arg)   = RealValue(abuf(1),45,6)*1E-5
    iexp     = IntegerValue(abuf(1),51,2)
    bstar_PC(arg)    = RealValue(abuf(1),54,6)*1E-5
    ibexp    = IntegerValue(abuf(1),60,2)
    elset_PC(arg)    = TRIM$(STR$(IntegerValue(abuf(1),66,3)))




    '{* Decode Card 2 *}
    xincl_PC(arg)    = RealValue(abuf(2),9,8)
    xnodeo_PC(arg)   = RealValue(abuf(2),18,8)
    eo_PC(arg)       = RealValue(abuf(2),27,7)*1E-7
    omegao_PC(arg)   = RealValue(abuf(2),35,8)
    xmo_PC(arg)      = RealValue(abuf(2),44,8)
    xno_PC(arg)      = RealValue(abuf(2),53,11)


    '{ period   := 1/xno; }
    '{* Convert TO proper units *}
    dummy=iexp
    xndd6o_PC(arg)   = xndd6o_PC(arg)*Power(10.0,dummy)
    dummy=ibexp
    bstar_PC(arg)    = bstar_PC(arg)*Power(10.0,dummy)/ae
    xnodeo_PC(arg)   = Radians(xnodeo_PC(arg))
    omegao_PC(arg)   = Radians(omegao_PC(arg))
    xmo_PC(arg)      = Radians(xmo_PC(arg))
    xincl_PC(arg)    = Radians(xincl_PC(arg))
    xno_PC(arg)      = xno_PC(arg)*twopi/xmnpda
    xndt2o_PC(arg)   = xndt2o_PC(arg)*twopi/(xmnpda*xmnpda)
    xndd6o_PC(arg)   = xndd6o_PC(arg)*twopi/Cube(xmnpda)
    '{* Determine whether Deep-Space Model is needed *}
    a1 = Power(xke/xno_PC(arg),tothrd)
    temp = 1.5*ck2*(3*SQR(COS(xincl_PC(arg)))-1)/Power(1 - eo_PC(arg)*eo_PC(arg),1.5)
    del1 = temp/(a1*a1)
    ao = a1*(1 - del1*(0.5*tothrd + del1*(1 + 134/81*del1)))
    delo = temp/(ao*ao)
    xnodp = xno_PC(arg)/(1 + delo)
    IF (twopi/xnodp >= 225) THEN
        ideep_PC(arg) = 1
    ELSE
        ideep_PC(arg) = 0
    END IF
    iflag_PC(arg) = 1

    ERASE abuf

    'msgbox  str$(xincl_PC(0))
END FUNCTION


FUNCTION TLEChecksum(cline AS STRING) AS STRING
    LOCAL i AS LONG, sum AS LONG, c AS STRING

    FOR i = 1 TO LEN(cline)
        c = MID$(cline, i, 1)
        IF c >= "0" AND c <= "9" THEN sum = sum + VAL(c)
        IF c = "-" THEN INCR sum
    NEXT

    FUNCTION = FORMAT$(sum MOD 10)
END FUNCTION

FUNCTION FormatExpField(eval AS DOUBLE) AS STRING
    LOCAL mant AS DOUBLE, eexp AS LONG

    IF eval = 0 THEN
        FUNCTION = "00000-0"
        EXIT FUNCTION
    END IF

    eexp = 0
    mant = eval

    DO WHILE ABS(mant) < 0.1
        mant = mant * 10
        DECR eexp
    LOOP

    DO WHILE ABS(mant) >= 1
        mant = mant / 10
        INCR eexp
    LOOP

    mant = mant * 100000

    FUNCTION = RIGHT$("00000" + FORMAT$(ABS(mant), "0"), 5) + _
               IIF$(eexp >= 0, "+", "-") + _
               RIGHT$("0" + FORMAT$(ABS(eexp), "0"), 1)
END FUNCTION




SUB RVtoElements(r AS Vec3, v AS Vec3, mu AS DOUBLE, _
                 a AS DOUBLE, e AS DOUBLE, inc AS DOUBLE, _
                 RAAN AS DOUBLE, argp AS DOUBLE, M AS DOUBLE)

    LOCAL h AS Vec3, n AS Vec3, evec AS Vec3
    LOCAL rmag AS DOUBLE, vmag AS DOUBLE, vmag2 AS DOUBLE
    LOCAL hmag AS DOUBLE, nmag AS DOUBLE
    LOCAL energy AS DOUBLE

    LOCAL temp AS Vec3
    LOCAL nu AS DOUBLE
    LOCAL EccAnom AS DOUBLE
    LOCAL pi AS DOUBLE

'WriteAudit 1, "mu = " + FORMAT$(mu, "###0.000")

    pi = 3.14159265358979##

    ' ----------------------------------------
    ' Magnitudes
    ' ----------------------------------------
    rmag = VecNorm(r)
    vmag = VecNorm(v)
    vmag2 = VecDot(v, v)     ' ? CRITICAL FIX (use squared magnitude)

    IF rmag <= 0## THEN EXIT SUB

    ' ----------------------------------------
    ' Angular momentum h = r × v
    ' ----------------------------------------
    CALL Cross(r, v, h)
    hmag = VecNorm(h)

    IF hmag <= 0## THEN EXIT SUB

    ' ----------------------------------------
    ' Inclination
    ' ----------------------------------------
    inc = ARCCOS(h.z / hmag)

    ' ----------------------------------------
    ' Node vector n = k × h
    ' ----------------------------------------
    n.x = -h.y
    n.y =  h.x
    n.z =  0##
    nmag = VecNorm(n)

    ' ----------------------------------------
    ' RAAN
    ' ----------------------------------------
    IF nmag > 1D-12 THEN
        RAAN = ARCCOS(n.x / nmag)
        IF n.y < 0## THEN RAAN = 2## * pi - RAAN
    ELSE
        RAAN = 0##
    END IF

    ' ----------------------------------------
    ' Eccentricity vector
    ' evec = (v×h)/mu - r/|r|
    ' ----------------------------------------
    CALL Cross(v, h, temp)
    CALL VecScale(temp, 1## / mu, temp)

    CALL VecScale(r, 1## / rmag, evec)
    CALL VecSub2(temp, evec, evec)

    e = VecNorm(evec)

    ' ----------------------------------------
    ' Argument of perigee
    ' ----------------------------------------
    IF nmag > 1D-12 AND e > 1D-8 THEN
        argp = ARCCOS(VecDot(n, evec) / (nmag * e))
        IF evec.z < 0## THEN argp = 2## * pi - argp
    ELSE
        argp = 0##
    END IF

    ' ----------------------------------------
    ' True anomaly
    ' ----------------------------------------
    IF e > 1D-8 THEN
        nu = ARCCOS(VecDot(evec, r) / (e * rmag))
        IF VecDot(r, v) < 0## THEN nu = 2## * pi - nu
    ELSE
        nu = 0##
    END IF

    ' ----------------------------------------
    ' Semi-major axis (FIXED)
    ' ----------------------------------------
    energy = vmag2 / 2## - mu / rmag   ' ? USE v^2, not |v|

    IF ABS(energy) > 1D-12 THEN
        a = -mu / (2## * energy)
    ELSE
        a = 0##
    END IF

    ' ----------------------------------------
    ' Mean anomaly
    ' ----------------------------------------
    IF e < 1## THEN
        EccAnom = 2## * ATN(SQR((1## - e) / (1## + e)) * TAN(nu / 2##))
        M = EccAnom - e * SIN(EccAnom)
    ELSE
        M = 0##
    END IF

    ' ----------------------------------------
    ' Debug (optional)
    ' ----------------------------------------
    WriteAudit 1, "rmag = " + FORMAT$(rmag, "###0.000")
    WriteAudit 1, "vmag = " + FORMAT$(vmag, "###0.000")
    WriteAudit 1, "energy = " + FORMAT$(energy, "###0.000000")

END SUB

FUNCTION StreakAvgEpoch AS DOUBLE
    LOCAL t AS LONG, sCnt AS LONG
    LOCAL JDTotal AS DOUBLE

    sCnt=0
    FOR t=1 TO gStreakTotalCnt
        IF StreakObs(t).UseObs=1 THEN
            INCR sCnt
            JDTotal=JDTotal+StreakObs(t).MidTimeJD
        END IF
    NEXT t
    IF sCnt<>0 THEN
        FUNCTION=jdTotal/sCnt
    ELSE
        FUNCTION = StreakObs(1).MidTimeJD
    END IF

    writeaudit 1, "cnt="+STR$(sCnt)+" avg="+STR$(jdTotal/sCnt)
END FUNCTION

SUB GetEpochState_old(SatelliteIndex AS LONG, epochJD AS DOUBLE, pPos0 AS vector, pVel0 AS vector)

    LOCAL t AS LONG

    iflag_PC(SatelliteIndex) = 1
    ConvertSatelliteData SatelliteIndex, 24
    SGP epochJD, pPos0, pVel0, 24
    ConvertSatelliteState pPos0, pVel0

END SUB





SUB ComputeOneResidual(cMethod AS LONG, i AS LONG, Atrk AS DOUBLE, Xtrk AS DOUBLE, Fitted AS LONG)

    LOCAL RA_pred AS DOUBLE, Dec_pred AS DOUBLE, Range_pred AS DOUBLE
    LOCAL dRA AS DOUBLE, dDec AS DOUBLE
    LOCAL mRA AS DOUBLE, mDec AS DOUBLE, mag AS DOUBLE
    LOCAL uRA AS DOUBLE, uDec AS DOUBLE
    LOCAL cosDecPred AS DOUBLE, cosDecObs AS DOUBLE

    '----------------------------------------
    ' Prediction
    '----------------------------------------
    IF cMethod = 1 THEN
        StreakExpectedPosJD(StreakObs(i).MidTimeJD, RA_pred, Dec_pred, Range_pred,Fitted)
        'StreakExpectedPosJD_old(StreakObs(i).MidTimeJD, RA_pred, Dec_pred, Range_pred)
        'WriteAudit 1, "Obs " + STR$(i) + _
        '      " RA_obs=" + STR$(StreakObs(i).MidRA) + _
        '      " Dec_obs=" + STR$(StreakObs(i).MidDec) + _
        '      " RA_pred=" + STR$(RA_pred) + _
         '     " Dec_pred=" + STR$(Dec_pred)+_
         '     "JD ="+ STR$(StreakObs(i).MidTimeJD)
    END IF

    cosDecPred = COS(Dec_pred * gDEG2RAD)
    cosDecObs  = COS(StreakObs(i).MidDec * gDEG2RAD)

    '----------------------------------------
    ' Residuals (arcsec)
    '----------------------------------------
    dRA = StreakObs(i).MidRA - RA_pred
    dRA = dRA - 360## * INT((dRA + 180##) / 360##)
    dRA = dRA * cosDecPred * 3600##

    dDec = (StreakObs(i).MidDec - Dec_pred) * 3600##

    '----------------------------------------
    ' Track direction (from observed streak)
    '----------------------------------------
    mRA = StreakObs(i).EndRA - StreakObs(i).StartRA
    mRA = mRA - 360## * INT((mRA + 180##) / 360##)
    mRA = mRA * cosDecObs * 3600##

    mDec = (StreakObs(i).EndDec - StreakObs(i).StartDec) * 3600##

    mag = SQR(mRA*mRA + mDec*mDec)

    IF mag > 1## THEN
        uRA = mRA / mag
        uDec = mDec / mag
    ELSE
        uRA = 0##
        uDec = 0##
    END IF

    '----------------------------------------
    ' Along-track / Cross-track residuals
    '----------------------------------------
    Atrk = dRA * uRA + dDec * uDec
    Xtrk = -dRA * uDec + dDec * uRA



END SUB





SUB ComputeRMSResidual(cMethod AS LONG, Fitted AS LONG, RMS AS DOUBLE)

    LOCAL i AS LONG
    LOCAL Atrk AS DOUBLE, Xtrk AS DOUBLE
    LOCAL sumsq AS DOUBLE
    LOCAL n AS LONG

    sumsq = 0##
    n = 0

    FOR i = 1 TO UBOUND(StreakObs)

        IF StreakObs(i).UseObs = 1 THEN

            CALL ComputeOneResidual(cMethod, i, Atrk, Xtrk, Fitted)

            sumsq = sumsq + Atrk*Atrk + Xtrk*Xtrk
            INCR n

        END IF

    NEXT i

    IF n > 0 THEN
        RMS = SQR(sumsq / n)
    ELSE
        RMS = 0##
    END IF

END SUB




' =============================================================
' Compute residual for a specific point on a streak
' pointType:  0 = Start,  1 = Mid,  2 = End
' =============================================================
SUB ComputeOneResidualForPoint(i AS LONG, pointType AS LONG, _
                               BYREF Atrk AS DOUBLE, BYREF Xtrk AS DOUBLE)

    LOCAL RA_pred AS DOUBLE, Dec_pred AS DOUBLE, Range_pred AS DOUBLE
    LOCAL dRA AS DOUBLE, dDec AS DOUBLE
    LOCAL mRA AS DOUBLE, mDec AS DOUBLE, mag AS DOUBLE
    LOCAL uRA AS DOUBLE, uDec AS DOUBLE
    LOCAL cosDecPred AS DOUBLE, cosDecObs AS DOUBLE
    LOCAL obsRA AS DOUBLE, obsDec, obsTime AS DOUBLE

    ' Get observed RA/Dec for the requested point
    SELECT CASE pointType
        CASE 0  ' Start
            obsRA  = StreakObs(i).StartRA
            obsDec = StreakObs(i).StartDec
            obsTime= StreakObs(i).StartTimeJD
        CASE 1  ' Mid (default, highest weight)
            obsRA  = StreakObs(i).MidRA
            obsDec = StreakObs(i).MidDec
            obsTime= StreakObs(i).MidTimeJD
        CASE 2  ' End
            obsRA  = StreakObs(i).EndRA
            obsDec = StreakObs(i).EndDec
            obsTime= StreakObs(i).EndTimeJD
    END SELECT

    ' Predict RA/Dec using current corrections
    StreakExpectedPosJD(ObsTime, RA_pred, Dec_pred, Range_pred,1)

    cosDecPred = COS(Dec_pred * gDEG2RAD)
    cosDecObs  = COS(obsDec * gDEG2RAD)

    ' RA residual in arcseconds (accounting for cos(dec))
    dRA = obsRA - RA_pred
    dRA = dRA - 360## * INT((dRA + 180##) / 360##)
    dRA = dRA * cosDecPred * 3600##

    dDec = (obsDec - Dec_pred) * 3600##

    writeaudit 1, "   dRA=" + FORMAT$(dRA,"#.000") + " dDec=" + FORMAT$(dDec,"#.000")

    ' Track direction vector from the full streak (Start ? End)
    ' This is the same for all three points on the same streak
    mRA = StreakObs(i).EndRA - StreakObs(i).StartRA
    mRA = mRA - 360## * INT((mRA + 180##) / 360##)
    mRA = mRA * cosDecObs * 3600##

    mDec = (StreakObs(i).EndDec - StreakObs(i).StartDec) * 3600##

    mag = SQR(mRA * mRA + mDec * mDec)

    IF mag > 1## THEN
        uRA  = mRA / mag
        uDec = mDec / mag
    ELSE
        uRA  = 0##
        uDec = 0##
    END IF
    writeaudit 1, "Obs " + STR$(i) + " PointType=" + STR$(pointType) + _
              " mRA=" + FORMAT$(mRA,"#.000") + " mDec=" + FORMAT$(mDec,"#.000") + _
              " mag=" + FORMAT$(mag,"#.000") + " uRA=" + FORMAT$(uRA,"#.000") + " uDec=" + FORMAT$(uDec,"#.000")

    ' Along-track and Cross-track residuals
    Atrk = dRA * uRA + dDec * uDec
    Xtrk = -dRA * uDec + dDec * uRA

END SUB




SUB StreakExpectedPosJD( jd AS DOUBLE, PredRA AS DOUBLE, PredDec AS DOUBLE, PredRange AS DOUBLE, Fitted AS LONG )
    LOCAL Usergeodetic AS vector
    LOCAL pPos, pVel, obsset, azmel AS vector
    LOCAL satelliteindex AS LONG
    LOCAL xincl_PC_save, xnodeo_PC_save, omegao_PC_save, xmo_PC_save, eo_PC_save, xno_pc_save AS DOUBLE

    satelliteindex = 0

    ' === 1. SAVE ORIGINALS (once per call) ===
    xincl_PC_save   = xincl_pc(satelliteindex)
    xnodeo_PC_save  = xnodeo_pc(satelliteindex)
    omegao_PC_save  = omegao_pc(satelliteindex)
    xmo_PC_save     = xmo_pc(satelliteindex)
    eo_PC_save      = eo_pc(satelliteindex)
    xno_pc_save     = xno_pc(satelliteindex)

    'writeaudit 1,"orbitals= "+STR$(r2d(xincl_pc(satelliteindex)))

    ' === 2. CLAMP (once per call) ===
    IF ABS(g_dM)  > 0.12## THEN g_dM  = SGN(g_dM)  * 0.12##
    IF ABS(g_dOm) > 0.008## THEN g_dOm = SGN(g_dOm) * 0.008##
    IF ABS(g_dw)  > 0.008## THEN g_dw  = SGN(g_dw)  * 0.008##
    IF ABS(g_di)  > 0.004## THEN g_di = SGN(g_di) * 0.004##
    IF ABS(g_de)  > 0.00005## THEN g_de = SGN(g_de) * 0.00005##
    g_da = 0.0##

   ' writeaudit 1, "Trial - g_dM=" + FORMAT$(g_dM * gRAD2DEG, "#.000") + " deg"

    ' === 3. Apply deltas ONCE ===
   IF Fitted = 1 THEN

    xincl_pc(satelliteindex)   = xincl_PC_save   + g_di
    xnodeo_pc(satelliteindex)  = xnodeo_PC_save  + g_dOm

    ' Lock shape (IMPORTANT)
    omegao_pc(satelliteindex)  = omegao_PC_save
    eo_pc(satelliteindex)      = eo_PC_save

    ' Phase correction
    xmo_pc(satelliteindex)     = xmo_PC_save + g_dM

    ' Mean motion unchanged
    xno_pc(satelliteindex)     = xno_pc_save

END IF

    LOCAL a AS DOUBLE
    LOCAL n AS DOUBLE

    ' Current semi-major axis from mean motion
    a = (xke / xno_pc_save) ^ tothrd

    ' Apply correction
    a = a + g_da

    ' Convert back to mean motion
    n = xke / (a ^ 1.5##)

    xno_pc(satelliteindex) = n


    ' === 4. ConvertSatelliteData (required, as you want) ===
    ConvertSatelliteData satelliteindex, 24

    'writeaudit 1, "After ConvertSatelliteData - xmo(active)=" + FORMAT$(xmo(satelliteindex), "#.000000")

    iflag(satelliteindex) = 1

    ' === 5. Propagate ===
    Usergeodetic.v1 = radians(gStreakLatitude)
    Usergeodetic.v2 = radians(gStreakLongitude)
    Usergeodetic.v3 = gStreakElevation * 0.001

    SGP jd, pPos, pVel, 24
    ConvertSatelliteState pPos, pVel
    CalculateRADec pPos, pVel, Usergeodetic, jd, obsset, 24
    calculateObs pPos, pVel, Usergeodetic, jd, azmel, 24

    ' === 6. Output + bias ===
    PredRA = degrees(obsset.v1)
    PredDec = degrees(obsset.v2)
    PredRange = azmel.v3

   ' writeaudit 1, "Before TopoJ2000 Ra/Dec=" + STR$(PredRA) + "/" + STR$(PredDec)
    PredRA = PredRA / 15.0
    TopoJ2000 PredRA, PredDec
    PredRA = PredRA * 15.0
   ' PredRA = PredRA - 0.20
   ' writeaudit 1, "After bias correction RA/Dec=" + STR$(PredRA) + "/" + STR$(PredDec)

    ' === 7. RESTORE ORIGINALS ===
    xincl_pc(satelliteindex)   = xincl_PC_save
    xnodeo_pc(satelliteindex)  = xnodeo_PC_save
    omegao_pc(satelliteindex)  = omegao_PC_save
    xmo_pc(satelliteindex)     = xmo_PC_save
    eo_pc(satelliteindex)      = eo_PC_save
    xno_pc(satelliteindex)     = xno_pc_save

    xincl(satelliteindex)   = xincl_PC_save
    xnodeo(satelliteindex)  = xnodeo_PC_save
    omegao(satelliteindex)  = omegao_PC_save
    xmo(satelliteindex)     = xmo_PC_save
    eo(satelliteindex)      = eo_PC_save
    xno(satelliteindex)     = xno_pc_save

    iflag(satelliteindex) = 1

END SUB



SUB JDToTLEEpoch(jd AS DOUBLE, BYREF YY AS LONG, BYREF DDD_frac AS DOUBLE)

    LOCAL Z AS LONG
    LOCAL F AS DOUBLE
    LOCAL A AS LONG, B AS LONG, C AS LONG, D AS LONG, E AS LONG
    LOCAL alpha AS LONG
    LOCAL day AS DOUBLE
    LOCAL month AS LONG, year AS LONG
    LOCAL doyr AS LONG
    LOCAL hour AS DOUBLE

    Z = FIX(jd + 0.5##)
    F = (jd + 0.5##) - Z

    IF Z < 2299161 THEN
        A = Z
    ELSE
        alpha = FIX((Z - 1867216.25##) / 36524.25##)
        A = Z + 1 + alpha - FIX(alpha / 4)
    END IF

    B = A + 1524
    C = FIX((B - 122.1##) / 365.25##)
    D = FIX(365.25## * C)
    E = FIX((B - D) / 30.6001##)

    day = B - D - FIX(30.6001## * E) + F

    IF E < 14 THEN
        month = E - 1
    ELSE
        month = E - 13
    END IF

    IF month > 2 THEN
        year = C - 4716
    ELSE
        year = C - 4715
    END IF

    ' --- Compute day of year ---
    doyr = DayOfYear2(year, month, FIX(day))

    ' fractional part of day
    hour = (day - FIX(day)) * 24.0##
    DDD_frac = doyr + hour / 24.0##

    YY = year MOD 100

END SUB

FUNCTION DayOfYear2(year AS LONG, month AS LONG, day AS LONG) AS LONG

    LOCAL i AS LONG
    LOCAL dimMonth() AS LONG

    DIM dimMonth(1 TO 12)

    dimMonth(1)=31 : dimMonth(2)=28 : dimMonth(3)=31
    dimMonth(4)=30 : dimMonth(5)=31 : dimMonth(6)=30
    dimMonth(7)=31 : dimMonth(8)=31 : dimMonth(9)=30
    dimMonth(10)=31 : dimMonth(11)=30 : dimMonth(12)=31

    ' Leap year
    IF (year MOD 4 = 0 AND year MOD 100 <> 0) OR (year MOD 400 = 0) THEN
        dimMonth(2) = 29
    END IF

    FOR i = 1 TO month - 1
        day = day + dimMonth(i)
    NEXT

    FUNCTION = day

END FUNCTION


SUB UpdateTLEEpoch(jd AS DOUBLE)

    LOCAL YY AS LONG
    LOCAL DDD_frac AS DOUBLE
    LOCAL epochStr AS STRING
    LOCAL line1 AS STRING

    CALL JDToTLEEpoch(jd, YY, DDD_frac)

    ' Format: YYDDD.DDDDDDDD
    epochStr = FORMAT$(YY, "00") + FORMAT$(DDD_frac, "000.00000000")

    line1 = StreakSatelliteData(2)

    ' Replace columns 19–32 (1-based indexing)
    MID$(line1, 19, 14) = epochStr

    ' Update checksum
    line1 = FixTLEChecksum(line1)

    StreakSatelliteData(2) = line1

END SUB



FUNCTION FixTLEChecksum(tline AS STRING) AS STRING

    LOCAL i AS LONG
    LOCAL sum AS LONG
    LOCAL ch AS STRING

    sum = 0

    FOR i = 1 TO 68
        ch = MID$(tline, i, 1)

        IF ch >= "0" AND ch <= "9" THEN
            sum = sum + VAL(ch)
        ELSEIF ch = "-" THEN
            sum = sum + 1
        END IF
    NEXT

    MID$(tline, 69, 1) = RIGHT$(STR$(sum MOD 10), 1)

    FUNCTION = tline

END FUNCTION



FUNCTION Normalize360(x AS DOUBLE) AS DOUBLE

    x = x - 360## * INT(x / 360##)

    IF x < 0## THEN x = x + 360##

    FUNCTION = x

END FUNCTION


FUNCTION StreaksUsed AS LONG
    LOCAL t, cnt AS LONG

    FOR t=1 TO gStreakCount
        IF StreakObs(t).UseObs<>0 THEN INCR cnt
    NEXT
    FUNCTION=cnt


END FUNCTION

SUB StreakOrbitalClear(complete AS LONG)

    IF complete=1 THEN
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEpochOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtInclOrig, "
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtRAANOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEccOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtArgPerOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMeanAnomOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMeanMotionOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtRevNumOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMM1Orig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMM2Orig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtBSTAROrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEphemerisOrig, ""
        CONTROL SET TEXT  hOrbitCreateTab, %streakTxtElsetOrig, ""
        g_di=0
        g_dOm=0
        g_dw=0
        g_dM=0
        g_de=0
        CONTROL SET TEXT hOrbitCreateTab, %streakFrmOrbit, "Orbital Elements"
    END IF

    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEpochNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtInclNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtRAANNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEccNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtArgPerNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMeanAnomNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMeanMotionNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtRevNumNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMM1New,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtMM2New,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtBSTARNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtEphemerisNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab, %streakTxtElsetNew,  ""
    CONTROL SET TEXT  hOrbitCreateTab,%streakTxtObsUsed,""
    CONTROL SET TEXT  hOrbitCreateTab,%streakTxtObsRejected,""
    CONTROL SET TEXT  hOrbitCreateTab,%streakTxtInitialRMS,""
    CONTROL SET TEXT  hOrbitCreateTab,%streakTxtPostResidual,""

END SUB




FUNCTION GetJDPrecise AS DOUBLE

    LOCAL pt AS IPOWERTIME
    LOCAL ft AS QUAD
    LOCAL seconds AS DOUBLE

    LET pt = CLASS "PowerTime"

    pt.Now        ' LOCAL time

    ft = pt.FileTime   ' 100-ns ticks since 1601

    ' Convert to seconds
    seconds = ft / 10000000##

    ' Convert FILETIME ? JD (still assumes UTC baseline!)
    FUNCTION = 2305813.5## + (seconds / 86400##)

END FUNCTION





'
