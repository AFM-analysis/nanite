"""nanite models summary

Usage
-----
Directives:

Table of all models available in nanite, including a texed model function

   .. nanite_model_table::


Documents all available models (for code reference)

   .. nanite_model_doc::

"""
from docutils.statemachine import ViewList
from docutils.parsers.rst import Directive
from sphinx.util.nodes import nested_parse_with_titles
from docutils import nodes

from nanite import model


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


class ModelDoc(Base):
    def generate_rst(self):
        rst = []

        keys = sorted(model.models_available.keys())

        for kk in keys:
            mod = model.models_available[kk]
            rst.append("")
            rst.append(".. _sec_ref_model_{}:".format(kk))
            rst.append("")
            rst.append(mod.model_name)
            rst.append("~" * len(mod.model_name))

            rst.append(".. csv-table::")
            rst.append("    :delim: tab")
            rst.append("")
            rst.append("    model key\t {}".format(mod.model_key))
            rst.append("    model name\t {}".format(mod.model_name))
            rst.append("    model location\t {}".format(mod.module.__name__))
            rst.append("")

            rst.append(".. automodule:: {}".format(mod.module.__name__))
            rst.append("    :members:")
            rst.append("    :undoc-members:")
            rst.append("    :exclude-members: model, residual, "
                       + "get_parameter_defaults, model_func")
            rst.append("")

        return rst


class ModelTable(Base):
    def generate_rst(self):
        rst = []

        rst.append(".. csv-table::")
        rst.append("    :header: model key, description, details")
        rst.append("    :delim: tab")
        rst.append("")

        keys = sorted(model.models_available.keys())

        for kk in keys:
            mod = model.models_available[kk]
            ref = "sec_ref_model_{}".format(kk)
            details = ":ref:`code reference <{}>`".format(ref)
            rst.append("    {}\t {}\t {}".format(kk, mod.model_name, details))

        rst.append("")

        return rst


def setup(app):
    app.add_directive('nanite_model_table', ModelTable)
    app.add_directive('nanite_model_doc', ModelDoc)
    return {'version': '0.1'}   # identifies the version of our extension
