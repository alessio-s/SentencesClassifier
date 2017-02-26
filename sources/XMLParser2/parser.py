import sys
import os
import xml.etree.ElementTree as ET
from string import punctuation

reload(sys)
sys.setdefaultencoding(sys.stdout.encoding)

DEBUG = False

def remove_punctuation(s):
    return ''.join(c for c in s if c not in punctuation)

def parse(xml_filename, txt_filename, labels_filename):
    tree = ET.parse(xml_filename).getroot()

    if not DEBUG:
        out_file = open(txt_filename, "w")
        labels_file = open(labels_filename, "w")

    for node in tree.iter('sentence'):
        attrib = node.attrib
        # Case of uncertain sentence
        if attrib['certainty'] == 'uncertain':
            if node.text is not None:
                if not DEBUG:
                    out_file.write(node.text)
                tmp = remove_punctuation(node.text).split(' ')
                if not (len(tmp) == 1 and tmp[0] == ''):
                    number_of_words = sum(x is not '' for x in remove_punctuation(node.text).split(' '))
                    for i in xrange(number_of_words):
                        if DEBUG:
                            sys.stdout.write('0 ')
                        else:
                            labels_file.write('0 ')

            # Iterate the children
            children = node.getchildren()
            for child in children:
                if child.tag == 'ccue':
                    if not DEBUG:
                        out_file.write(child.text)
                    tmp = remove_punctuation(child.text).split(' ')
                    if not (len(tmp) == 1 and tmp[0] == ''):
                        number_of_words = sum(x is not '' for x in remove_punctuation(child.text).split(' '))
                        for i in xrange(number_of_words):
                            if DEBUG:
                                sys.stdout.write('1 ')
                            else:
                                labels_file.write('1 ')
                    if child.tail is not None:
                        if not DEBUG:
                            out_file.write(child.tail)
                        tmp = remove_punctuation(child.tail).split(' ')
                        if not (len(tmp) == 1 and tmp[0] == ''):
                            number_of_words = sum(x is not '' for x in remove_punctuation(child.tail).split(' '))
                            for i in xrange(number_of_words):
                                if DEBUG:
                                    sys.stdout.write('0 ')
                                else:
                                    labels_file.write('0 ')
            if not DEBUG:
                out_file.write('\n')
                labels_file.write('\n')

        # Case of certain sentence
        else:
            if node.text is not None:
                if not DEBUG:
                    out_file.write(node.text)
                tmp = remove_punctuation(node.text).split(' ')
                if not (len(tmp) == 1 and tmp[0] == ''):
                    number_of_words = sum(x is not '' for x in remove_punctuation(node.text).split(' '))
                    for i in xrange(number_of_words):
                        if DEBUG:
                            sys.stdout.write('0 ')
                        else:
                            labels_file.write('0 ')
                if not DEBUG:
                    out_file.write('\n')
                    labels_file.write('\n')
        if DEBUG:
            sys.stdout.flush()
    if DEBUG:
        sys.stdout.flush()
    else:
        out_file.close()
        labels_file.close()


if __name__ == "__main__":
    xml_filename = "/home/alessio/PycharmProjects/XMLParser2/xml_files/task1_train_wikipedia_rev2.xml"
    txt_filename = "/home/alessio/PycharmProjects/XMLParser2/parsed_files/wikipedia.txt"
    labels_filename = "/home/alessio/PycharmProjects/XMLParser2/parsed_files/wikipedia_labels.txt"
    parse(xml_filename, txt_filename, labels_filename)