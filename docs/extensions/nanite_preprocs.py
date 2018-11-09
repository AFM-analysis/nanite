"""nanite preprocessing summary

Usage
-----
Directives:

Table of all preprocessing methods available in nanite

   .. nanite_preproc_table::

"""
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes

from nanite import preproc


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

        keys = preproc.available_preprocessors

        rst.append(".. csv-table::")
        rst.append("    :header: preprocessor key, description, details")
        rst.append("    :delim: tab")
        rst.append("")

        for kk in keys:
            ref = "nanite.preproc.IndentationPreprocessor.{}".format(kk)
            details = ":func:`code reference <{}>`".format(ref)
            method = getattr(preproc.IndentationPreprocessor, kk)
            name = method.__doc__.split("\n\n")[0].strip()
            rst.append("    {}\t {}\t {}".format(kk, name, details))

        rst.append("")

        return rst


def setup(app):
    app.add_directive('nanite_preproc_table', PreprocTable)
    return {'version': '0.1'}   # identifies the version of our extension
