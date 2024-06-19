import pandas as pd
class LogtoCsv:
    def __init__(self, filename_cmx, dir, fileout, filename_pwr, offsettimepwr):
        self.fileid_cmx = filename_cmx
        self.dir = dir
        self.fileout = fileout
        self.fileid_pwr = filename_pwr
        self.fs = 5000
        self.meanreso = 2000
        self.offsettimepwr = offsettimepwr*self.fs #0.883*self.fs # nr of lines to be removed

    def readfile(self,fileid):
        file_open = open(fileid, 'r')
        read_lines = file_open.readlines()
        #print(self.fileid)
        return read_lines

    def writefile(self):
        filenameout = self.dir + self.fileout
        file_write = open(filenameout, 'w')
        return file_write

    def checknav(self,chknav):
        if chknav.find('NAV') != -1:
            return 'NAV'
        else:
            return float(chknav)

    def readpower(self,readlines, timestamp, meanreso, nrlines):
        start = int(float(timestamp)*self.fs) - int(meanreso/2) + 1 + int(self.offsettimepwr)
        if start < 1:
            start = 1
        #stop = start + int(meanreso/2)
        meanpwr = 0
        if (start + meanreso) < nrlines:
            for line in range(meanreso):
                index = line + start
                meanpwr = meanpwr + float(readlines[index].split(',')[2])
            meanpwr =round(meanpwr/meanreso,2)
            return meanpwr
        else:
            return 'NAV'

    def rmvLastrow(self):
        filenameout = self.dir + self.fileout
        df = pd.read_csv(filenameout)
        df = df.drop(df.index[-1])
        df.to_csv(filenameout, index=False)

    def LogtoCsvDef(self):
        readlines = self.readfile(self.fileid_cmx) #read log cmx
        linelen = len(readlines)
        lineIdx = 0
        newline = False

        readlines_pwr = self.readfile(self.fileid_pwr) # read log power
        nrlines_pwr = len(readlines_pwr)

        # private parameters #shold not be changed
        loglength = 25 # a complete log length
        NRULpwr = 0 # UE UL power
        NRDLpwr = 10.1 # NR Cell power
        LTEULpwr = 0 # UE UL power
        LTEDLpwr = -52.2 # lte cell power
        #write file
        f = self.writefile()
        f.write('time,DL_LteThp(bps),DL_NrDlThp(bps),DL_LteAck%,DL_LteNack%,DL_LteDtx%,DL_LteBLER%,DL_LteThp%,DL_NrAck%,DL_NrNack%,DL_NrDtx%,DL_NrBLER%,DL_NrThp%,UL_LteAck%,UL_LteNack%,UL_LteDtx%,UL_LteBLER%,UL_NrAck%,UL_NrNack%,UL_NrDtx%,UL_NrBLER%,UL_LteThp%,UL_LteTph(bps),UL_NrThp%,UL_NrTph(bps),Lte_RSRP(dBm),Lte_RSRQ(dBm),NR_RSRP(dBm),NR_RSRQ(dBm),NR_SINR(dBm),PowerMeas(mW),NRULpwr(dBm), NRDLpwr(dBm), LTEULpwr(dBm), LTEDLpwr(dBm) \n')

        while (lineIdx+loglength) < linelen:

            if readlines[lineIdx].find('%') != -1 and (lineIdx+loglength) > linelen:
                break

            elif readlines[lineIdx].find('%') != -1: # if find time stamp
                timestamp = readlines[lineIdx].split('%')[1]
                timestamp = str(round(float(timestamp),4)) #format to 4 decimal
                f.write(timestamp + ', ')

            elif readlines[lineIdx].find('FETCh:SIGN:MEAS:BLER:ABSolute?') != -1: #DL BLER
                lineIdx += 2
                if readlines[lineIdx].find('NR') != -1: # with ENDC
                    readData = readlines[lineIdx].replace('\'','').replace(']','').split(",")
                    DL_LteThp_bps = int(readData[5])
                    f.write(str(DL_LteThp_bps) + ', ')
                    DL_NrDlThp_bps = int(readData[10])
                    f.write(str(DL_NrDlThp_bps) + ', ')
                else: # with LTE only
                    readData = readlines[lineIdx].replace('\'', '').replace(']', '').split(",")
                    DL_LteThp_bps = int(readData[5])
                    f.write(str(DL_LteThp_bps) + ', ')
                    f.write('- , ')
            elif readlines[lineIdx].find('FETCh:SIGN:MEAS:BLER:RELative?') != -1: #DL BLER
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace(']', '').split(',')
                DL_LteAck = self.checknav(readData[2]) #LTE
                f.write(str(DL_LteAck) + ', ')
                DL_LteNack = self.checknav(readData[3])
                f.write(str(DL_LteNack) + ', ')
                DL_LteDtx = self.checknav(readData[4])
                f.write(str(DL_LteDtx) + ', ')
                DL_LteBLER = self.checknav(readData[5])
                f.write(str(DL_LteBLER) + ', ')
                DL_LteThp = self.checknav(readData[6])
                f.write(str(DL_LteThp) + ', ')
                if readlines[lineIdx].find('NR') != -1: # with ENDC
                    DL_NrAck = self.checknav(readData[8]) #NR
                    f.write(str(DL_NrAck) + ', ')
                    DL_NrNack = self.checknav(readData[9])
                    f.write(str(DL_NrNack) + ', ')
                    DL_NrDtx = self.checknav(readData[10])
                    f.write(str(DL_NrDtx) + ', ')
                    DL_NrBLER = self.checknav(readData[11])
                    f.write(str(DL_NrBLER) + ', ')
                    DL_NrThp = self.checknav(readData[12])
                    f.write(str(DL_NrThp) + ', ')
                else: # LTE only
                    f.write('- ,- ,- ,- ,- , ')

            elif readlines[lineIdx].find('FETCh:SIGNaling:MEASurement:BLER:UL:RELative?') != -1:
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace(']', '').split(',')
                #print(readData)
                UL_LteAck = self.checknav(readData[3])  # LTE
                f.write(str(UL_LteAck) + ', ')
                UL_LteNack = self.checknav(readData[2])
                f.write(str(UL_LteNack) + ', ')
                UL_LteDtx = self.checknav(readData[4])
                f.write(str(UL_LteDtx) + ', ')
                UL_LteBLER = self.checknav(readData[5])
                f.write(str(UL_LteBLER) + ', ')
                if readlines[lineIdx].find('NR') != -1:  # with ENDC
                    UL_NrAck = self.checknav(readData[8])
                    f.write(str(UL_NrAck) + ', ')
                    UL_NrNack = self.checknav(readData[7])
                    f.write(str(UL_NrNack) + ', ')
                    UL_NrDtx = self.checknav(readData[9])
                    f.write(str(UL_NrDtx) + ', ')
                    UL_NrBLER = self.checknav(readData[10])
                    f.write(str(UL_NrBLER) + ', ')
                else: #LTE only
                    f.write('- ,- ,- ,- , ')

            elif readlines[lineIdx].find('FETCh:SIGNaling:MEASurement:BLER:UL:THRoughput?') != -1:
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace(']', '').split(',')
                UL_LteThp = self.checknav(readData[2])  # LTE
                f.write(str(UL_LteThp) + ', ')
                UL_LteTph_bps = self.checknav(readData[3])  # LTE
                f.write(str(UL_LteTph_bps) + ', ')
                if readlines[lineIdx].find('NR') != -1:  # with ENDC
                    UL_NrThp = self.checknav(readData[6])
                    f.write(str(UL_NrThp) + ', ')
                    UL_NrThp_bps = self.checknav(readData[7])
                    f.write(str(UL_NrThp_bps) + ', ')
                else: #LTE only
                    f.write('- ,- , ')

            elif readlines[lineIdx].find('SENSe:DUT:MODem1:MEASurement:REPort?') != -1:
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace(']', '').split(',')

                if readlines[lineIdx].find('NR') != -1:  # with ENDC
                    Lte_RSRP = self.checknav(readData[6])  # LTE
                    f.write(str(Lte_RSRP) + ', ')
                    Lte_RSRQ = self.checknav(readData[9])  # LTE
                    f.write(str(Lte_RSRQ) + ', ')

                    NR_RSRP = self.checknav(readData[31])
                    f.write(str(NR_RSRP) + ', ')
                    NR_RSRP = self.checknav(readData[34])
                    f.write(str(NR_RSRP) + ', ')
                    NR_SINR = self.checknav(readData[37])
                    f.write(str(NR_SINR) + ', ')
                else:  # LTE only BECAREFULL it has different format, so different location (index)
                    #print(readData[18])
                    if readData[18].find('NAV') != -1:
                        Lte_RSRP = self.checknav(readData[6])  # LTE
                        f.write(str(Lte_RSRP) + ', ')
                        Lte_RSRQ = self.checknav(readData[9])  # LTE
                        f.write(str(Lte_RSRQ) + ', ')
                        f.write('- ,- ,- , ')  # blanks for NR
                    else:
                        Lte_RSRP = self.checknav(readData[18])  # LTE
                        f.write(str(Lte_RSRP) + ', ')
                        Lte_RSRQ = self.checknav(readData[22])  # LTE
                        f.write(str(Lte_RSRQ) + ', ')
                        f.write('- ,- ,- , ') # blanks for NR

                meanPwr = self.readpower(readlines_pwr,timestamp,self.meanreso,nrlines_pwr) # run mean power
                if meanPwr == 'NAV': # if it is NAV, delete the row and exit while loop
                    f.close()
                    self.rmvLastrow()
                    break
                f.write(str(meanPwr) + ', ' + str(NRULpwr) + ', ' + str(NRDLpwr) + ', ' + str(LTEULpwr) + ', ' + str(LTEDLpwr) + '\n')
            # if there are changes in power setting
            elif readlines[lineIdx].find('CONFigure:SIGNaling:NRADio:CELL:POWer:CONTrol:TPControl:CLOop:TPOWer?') != -1 and lineIdx > 9: #NR UE UL power
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace('[', '').replace(']', '').replace('\n', '').split(",")
                #print(readData, ',lineid=', lineIdx)
                NRULpwr = int(self.checknav(readData[0]))

            elif readlines[lineIdx].find('CONFigure:SIGNaling:LTE:CELL:POWer:CONTrol:TPControl:CLOop:TPOWer?') != -1 and lineIdx > 12: #Lte UE UL power
                lineIdx += 2
                readData = readlines[lineIdx].replace('\'', '').replace('[', '').replace(']', '').replace('\n', '').split(",")
                #print(readData,',lineid=',lineIdx)
                LTEULpwr = int(self.checknav(readData[0]))



            lineIdx += 1
        print('Log to CSV is done!')
        #f.close()


            #lineIdx += 1

        #print(lineIdx)



