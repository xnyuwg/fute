import logging
import random
from typing import List, Tuple
from basic_processor import BasicProcessor
from core.conf.global_config_manager import GlobalConfigManager
from core.utils.util_data import UtilData


class MATRESRelation:
    """ contains the content of only one relation """
    def __init__(self, element):
        """ init one relation of MATRES """
        self.raw_element = element
        self.label = element.attrib['LABEL']
        self.sentdiff = int(element.attrib['SENTDIFF'])
        self.docid = element.attrib['DOCID']
        self.source = element.attrib['SOURCE']
        self.target = element.attrib['TARGET']
        self.source_sentence = int(element.attrib['SOURCE_SENTID'])
        self.target_sentence = int(element.attrib['TARGET_SENTID'])
        self.data = element.text.strip().split()
        self.token = []
        self.lemma = []
        self.part_of_speech = []
        self.position = []
        self.length = len(self.data)
        self.event_ix = []
        self.text = ""
        self.event_offset = []

        is_start = True
        for i, d in enumerate(self.data):
            tmp = d.split('///')
            self.part_of_speech.append(tmp[-2])
            self.position.append(tmp[-1])
            self.token.append(tmp[0])
            self.lemma.append(tmp[1])
            if is_start:
                is_start = False
            else:
                self.text += " "
            if tmp[-1] == 'E1':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            elif tmp[-1] == 'E2':
                self.event_ix.append(i)
                self.event_offset.append(len(self.text))
            self.text += tmp[0]
        assert len(self.event_ix) == 2

        self.showed_token = self.token.copy()
        for i, ix in enumerate(self.event_ix):
            self.showed_token[ix] = "[Event " + str(i + 1) + ": " + self.showed_token[ix] + "]"
        self.showed_text = " ".join(self.showed_token)

    def print_all(self) -> str:
        """ print all attributes """
        res = "MATRESRelation:\n"
        res += ("label:{}\n".format(self.label))
        res += ("sentdiff:{}\n".format(self.sentdiff))
        res += ("docid:{}\n".format(self.docid))
        res += ("source:{}\n".format(self.source))
        res += ("target:{}\n".format(self.target))
        res += ("data:{}\n".format(self.data))
        res += ("token:{}\n".format(self.token))
        res += ("lemma:{}\n".format(self.lemma))
        res += ("part_of_speech:{}\n".format(self.part_of_speech))
        res += ("position:{}\n".format(self.position))
        res += ("length:{}\n".format(self.length))
        res += ("event_ix:{}\n".format(self.event_ix))
        res += ("text:{}\n".format(self.text))
        res += ("event_offset:{}\n".format(self.event_offset))
        res += ("showed_token:{}\n".format(self.showed_token))
        res += ("showed_text:{}\n".format(self.showed_text))
        return res

    def print_for_show(self) -> str:
        """ print showed attributes """
        res = ("showed_text: {} # label: {}. ".format(self.showed_text, self.label))
        return res


class MATRESProcessor(BasicProcessor):
    def __init__(self):
        """ init the MATRES path and read the raw data """
        super().__init__()
        self.MATRES_path = GlobalConfigManager.get_dataset_path('MATRES')
        logging.info("MATRES Path: {}".format(self.MATRES_path))

        self.trainset_xml = self.MATRES_path / "trainset-temprel.xml"
        self.testset_xml = self.MATRES_path / "testset-temprel.xml"
        logging.debug("trainset_xml: {}".format(self.trainset_xml))
        logging.debug("testset_xml: {}".format(self.testset_xml))
        # read data
        logging.info("reading data...")
        self.trainset_relations = self.read_xml_file(self.trainset_xml)
        self.testset_relations = self.read_xml_file(self.trainset_xml)
        self.trainset_docid_set = set([relation.docid for relation in self.trainset_relations])
        self.testset_docid_set = set([relation.docid for relation in self.testset_relations])

    def read_xml_file(self, file_name) -> List[MATRESRelation]:
        """ read MATRES .xml file, return the relation list. The relation is also a object containing the content. """
        root = UtilData.read_raw_xml_file(self.trainset_xml)
        relations = []
        for element in root:
            relation = MATRESRelation(element)
            relations.append(relation)
        # for debug
        logging.debug("randomly print one of file: {}".format(file_name))
        logging.debug(random.choice(relations).print_all())
        return relations

    def random_print_relation(self):
        """ print a random relation, only available with DEBUG log level"""
        print(random.choice(self.trainset_relations).print_all())

    def random_print_relation_in_one_docment(self, print_all=False):
        """ print all relations of a random document, only available with DEBUG log level"""
        docid = random.choice(list(self.trainset_docid_set))
        print("randomly print relations of docid {}:".format(docid))
        for relation in self.trainset_relations:
            if relation.docid == docid:
                print("---------- docid {} sentence {}-{}: -----------".format(docid, relation.source_sentence, relation.target_sentence))
                if print_all:
                    print(relation.print_all())
                else:
                    print(relation.print_for_show())

    def get_relations_object(self) -> Tuple[List[MATRESRelation], List[MATRESRelation]]:
        """ return the list of relations object of MATRESRelation """
        return self.trainset_relations, self.testset_relations


if __name__ == "__main__":
    # only for test
    m = MATRESProcessor()

