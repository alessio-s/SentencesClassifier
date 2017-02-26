import sys
import os
import xml.etree.ElementTree as ET

reload(sys)
sys.setdefaultencoding(sys.stdout.encoding)

DEBUG = False

def parse(xml_filename, txt_filename, labels_filename):
    tree = ET.parse(xml_filename).getroot()

    if not DEBUG:
        out_file = open(txt_filename, "w")
        labels_file = open(labels_filename, "w")

    for node in tree.iter('sentence'):
        attrib = node.attrib
        #Case of uncertain sentence
        if attrib['certainty'] == 'uncertain':
            children = node.getchildren()
            if node.text is not None:
                if DEBUG:
                    sys.stdout.write(node.text)
                else:
                    out_file.write(node.text)

            for child in children:
                if DEBUG:
                    sys.stdout.write(child.text)
                else:
                    out_file.write(child.text)
                if child.tail is not None:
                    if DEBUG:
                        sys.stdout.write(child.tail)
                    else:
                        out_file.write(child.tail)
            labels_file.write('1\n')

        # Case of certain sentence
        else:
            if node.text is not None:
                if DEBUG:
                    sys.stdout.write(node.text)
                else:
                    out_file.write(node.text)
                    labels_file.write('0\n')
        if DEBUG:
            sys.stdout.write('\n')
        else:
            out_file.write('\n')

    if DEBUG:
        sys.stdout.flush()
    else:
        out_file.close()
        labels_file.close()


if __name__ == "__main__":
    xml_filename = "/home/alessio/PycharmProjects/XMLParser/xml_files/task1_train_wikipedia_rev2.xml"
    txt_filename = "/home/alessio/PycharmProjects/XMLParser/parsed_files/wikipedia.txt"
    labels_filename = "/home/alessio/PycharmProjects/XMLParser/parsed_files/wikipedia_labels.txt"
    parse(xml_filename, txt_filename, labels_filename)