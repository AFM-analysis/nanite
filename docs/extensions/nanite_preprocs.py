"""nanite preprocessing summary

Usage
-----
Directives:

Table of all preprocessors available in nanite

   .. nanite_preproc_table::

Table of all POC methods in nanite

   .. nanite_preproc_poc_table::

"""
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes

from nanite import poc, preproc


class Base(Directive):
    required_arguments = 0
    optional_arguments = 0

    def generate_rst(self):
        pass

    def run(self):
        rst = self.generate_rst()

        vl = ViewList(rst, "fakefile.rst")
        # Create a node.
        node = nodes.section()
        node.document = self.state.document
        # Parse the rst.
        nested_parse_with_titles(self.state, vl, node)
        return node.children


class PreprocTable(Base):
    def generate_rst(self):
        rst = []

        rst.append(".. csv-table::")
        rst.append("    :header: preprocessor key, description, details")
        rst.append("    :delim: tab")
        rst.append("")

        for pp in preproc.PREPROCESSORS:
            ref = f"nanite.preproc.preproc_{pp.identifier}"
            details = ":func:`code reference <{}>`".format(ref)
            rst.append(f"    {pp.identifier}\t {pp.name}\t {details}")

        rst.append("")

        return rst


class PreprocPOCTable(Base):
    def generate_rst(self):
        rst = []

        pocs = poc.POC_METHODS

        rst.append(".. csv-table::")
        rst.append("    :header: POC method, description, details")
        rst.append("    :delim: tab")
        rst.append("")

        for pp in pocs:
            ref = f"nanite.poc.{pp.__name__}"
            details = ":func:`code reference <{}>`".format(ref)
            method = pp.identifier
            name = pp.name
            rst.append(f"    {method}\t {name}\t {details}")

        rst.append("")

        return rst


def setup(app):
    app.add_directive('nanite_preproc_table', PreprocTable)
    app.add_directive('nanite_preproc_poc_table', PreprocPOCTable)
    return {'version': '0.2'}   # identifies the version of our extension
