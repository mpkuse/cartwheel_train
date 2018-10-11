import editor # Needed for InteractiveLogger
import os


class InteractiveLogger:
    def __init__(self, LOG_DIR ):
        self.LOG_DIR = LOG_DIR
        self.string_data = []

        print '[InteractiveLogger] LOG_DIR = ', LOG_DIR
        print '[InteractiveLogger] mkdir -p ', LOG_DIR
        os.makedirs( LOG_DIR )

    def add_linetext( self, line ):
        self.string_data.append( line )

    def add_file( self, fname, txt ):
        print '[InteractiveLogger] Write file : ', self.LOG_DIR+"/"+fname
        text_file = open(self.LOG_DIR+"/"+fname, "w")
        text_file.write(txt)
        text_file.close()


    def fire_editor( self ):
        init_content = '#-------------------\n# %s\n#----------------\n### Initial Content\n%s\n### Notes here' %(self.LOG_DIR, '\n'.join( self.string_data) )

        result = editor.edit(contents=init_content )

        print '[InteractiveLogger] Write file : ', self.LOG_DIR+"/README.md"
        text_file = open(self.LOG_DIR+"/README.md", "w")
        text_file.write(result)
        text_file.close()


    def dir( self ):
        return self.LOG_DIR
