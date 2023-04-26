#!/usr/bin/env python3

import os

import numpy
from numpy.distutils.misc_util import Configuration

# Standard library imports
from os.path import join as pjoin, dirname
from distutils.dep_util import newer_group
from distutils.errors import DistutilsError

from numpy.distutils.misc_util import appendpath
from numpy.distutils import log


def generate_a_pyrex_source(self, base, ext_name, source, extension):
    ''' Monkey patch for numpy build_src.build_src method
    Uses Cython instead of Pyrex.
    Assumes Cython is present
    '''
    if self.inplace:
        target_dir = dirname(base)
    else:
        target_dir = appendpath(self.build_src, dirname(base))
    target_file = pjoin(target_dir, ext_name + '.c')
    depends = [source] + extension.depends
    if self.force or newer_group(depends, target_file, 'newer'):
        import Cython.Compiler.Main
        log.info("cythonc:> %s" % (target_file))
        self.mkpath(target_dir)
        options = Cython.Compiler.Main.CompilationOptions(
            defaults=Cython.Compiler.Main.default_options,
            include_path=extension.include_dirs,
            output_file=target_file)
        cython_result = Cython.Compiler.Main.compile(source,
                                                   options=options)
        if cython_result.num_errors != 0:
            raise DistutilsError("%d errors while compiling %r with Cython" \
                  % (cython_result.num_errors, source))
    return target_file


from numpy.distutils.command import build_src
build_src.build_src.generate_a_pyrex_source = generate_a_pyrex_source

def configuration(parent_package=None, top_path=None):
    config = Configuration(
        package_name = "emi",
        parent_name = parent_package,
        package_path = top_path,
    )
    libraries = []
    if os.name == "posix":
        libraries.append("m")
    config.add_extension(
        "_expected_mutual_info_fast",
        sources = ["_expected_mutual_info_fast.pyx"],
        include_dirs = [numpy.get_include()],
        #libraries = libraries,
    )
    print(config)
    return config
        

if __name__ == "__main__":
    from numpy.distutils.core import setup
    
    setup(**configuration().todict())
