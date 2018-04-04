#!/usr/bin/env python3
# course: TCSS555
# User profile in social media - ensemble
# date: 10/10/2017
# name: Team 4 - Iris Xia
# description: Executable Python file to reading data from inputfile
# and output the prediction into outputfile
import sys, getopt
import os
import tcss555_ensemble

# -----------------------------------------------
# Method for reading user input and output result
# -----------------------------------------------


def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv,'hi:o:',['ifile="','ofile='])
    except getopt.GetoptError:
        print ('error:\ntcss555.py -i <inputfile> -o <outputfile>\n')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('======\nusage:\n======\ntcss555.py -i <inputfile> -o <outputfile>\n')
            sys.exit()
        elif opt in ('-i', '--ifile'):
            inputfile = arg
        elif opt in ('-o', '--ofile'):
            outputfile = arg
            if not os.path.exists(outputfile):
                os.makedirs(outputfile)

    if inputfile != '' and outputfile != '':
        print ('Input file is :', inputfile)
        print ('Output file is :', outputfile,'\n')

        tcss555_ensemble.ensemble(inputfile, outputfile)
        #relation_gender.predictGender(inputfile)


main(sys.argv[1:])





