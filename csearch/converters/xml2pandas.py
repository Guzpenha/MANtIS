import xml.etree.ElementTree as ET
import pandas as pd


class XML2Pandas:

    def __init__(self, filename):
        data = ET.parse(filename)
        self.root = data.getroot()

    def parse_root(self, root):
        """Return a list of dictionaries from the text and attributes of the
        children under this XML root."""
        return [child.attrib for child in root.getchildren()]

    def convert(self):
        """ Initiate the root XML, parse it, and return a dataframe"""
        structure_data = self.parse_root(self.root)
        return pd.DataFrame(structure_data)
