## Given a tensorflow frozen graph (.pb) convert to nvidia's .uff



import argparse
import os.path
import TerminalColors
tcol = TerminalColors.bcolors()

import uff

if __name__ == '__main__':
    #---
    # Parse Arguments
    parser = argparse.ArgumentParser(description='Convert the Tensorflow Frozen protobuf (.pb) to Nvidia UFF Format')
    parser.add_argument('--frozenmodel', '-pb', type=str, required=True, help='The inputpath of Tensorflow frozen (.pb)')
    args = parser.parse_args()
    args.frozenmodel = args.frozenmodel.strip()


    #--- Files Exists?
    if not os.path.isfile(args.frozenmodel) :
        print tcol.FAIL, 'FILE NOT FOUND: ', args.frozenmodel, tcol.ENDC
        quit()
    print tcol.OKGREEN, 'Input TF-Frozen-Graph: ',tcol.ENDC , args.frozenmodel

    #--- has extension as .pb ?
    if not ( args.frozenmodel.split( '.')[-1] == 'pb' ):
        print tcol.FAIL, 'The input file need to be a .pb file. You supplied: ', args.frozenmodel, tcol.ENDC
        print 'QUIT...'
        quit()


    #--- Output path
    output_uff_fullfname = '.'.join( args.frozenmodel.split( '.')[:-1] )+'.uff'
    print tcol.BOLD, 'The UFF will be written to: ', tcol.ENDC, output_uff_fullfname

    if os.path.isfile(output_uff_fullfname) :
        print tcol.WARNING, "This file already exists", tcol.ENDC
        if raw_input("overwrite? (y/n)") is not 'y':
            print 'QUIT...'
            quit()

    #---
    # uff.from_tensorflow_frozen_model
    uff.from_tensorflow_frozen_model( frozen_file=args.frozenmodel, output_filename=output_uff_fullfname )
