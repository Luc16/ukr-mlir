# Tell Tablegen to use Ukernel.td as input.
set(LLVM_TARGET_DEFINITIONS Ukernel.td)

# Ask Tablegen to generate op declarations and definitions from ODS.
mlir_tablegen(Ukernel.h.inc -gen-op-decls)
mlir_tablegen(Ukernel.cpp.inc -gen-op-defs)

# Add a CMakeTarget we can depend on to ensure the generation happens before the compilation.
add_public_tablegen_target(UkernelDialectIncGen)

# Don't forget to generate the documentation, this will produce a
# UkernelDialect.md under docs/Dialects
add_mlir_doc(Ukernel UkernelDialect docs/Dialects/ -gen-op-doc)
