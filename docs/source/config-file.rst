The configuration file
=======================


The YAML format
---------------
The configuration file is the primary model of user input to the ``MF6model`` and ``MFnwtModel`` classes. Input is specified in the `yaml format`_, which can be thought of as a serialized python dictionary with some additional features, including the ability to include comments. Instead of curly brackets (as in `JSON`_), white space indentation is used to denote different levels of the dictionary. Value can generally be entered more or less as they are in python, except that dictionary keys (strings) don't need to be quoted. Numbers are parsed as integers or floating point types depending whether they contain a decimal point. Values in square brackets are cast into python lists; curly brackets can also be used to denote dictionaries instead of white space. Comments are indicated with the `#` symbol, and can be placed on the same line as data, as in python.

Modflow-setup uses the `pyyaml`_ package to parse the configuration file into the ``cfg`` dictionary attached to a model instance. The methods attached to ``MF6model``, ``MFnwtModel`` and ``MFsetupMixin`` then use the information in the ``cfg`` dictonary to set up various aspects of the model.


.. toctree::
   :maxdepth: 1
   :caption: Example problems

   Configuration file structure <config-file-structure>
   Configuration defaults <config-file-defaults>


Some additional notes on YAML
---------------------------------------
* quotes are optional for strings without special meanings. See `this reference`_ for more details.
* (``None`` and ``'none'``) (``'None'`` and ``'none'``) are parsed as strings (``'None'`` and ``'none'``)
* null is parsed to a ``NoneType`` instance (``None``)
* numbers in exponential format need a decimal place and a sign for the exponent to be parsed as floats.
  For example, as of pyyaml 5.3.1:

    * ``1e5`` parses to ``'1e5'``
    * ``1.e5`` parses to ``'1.e5'``
    * ``1.e+5`` parses to ``1.e5`` (a float)












.. _JSON: https://www.json.org/json-en.html
.. _pyyaml: https://pyyaml.org/
.. _this reference: http://blogs.perl.org/users/tinita/2018/03/strings-in-yaml---to-quote-or-not-to-quote.html
.. _yaml format: https://yaml.org/
