# Outside transform, this should be `add_mlir_library`.
add_mlir_library(
  # Library called SGemm.
  UkernelDialect

  # Built from the following source files.
  Ukernel.cpp
  # SplitToShards.cpp
  # LowerToBLAS.cpp

  # Make includes visible without top-level path.
  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include

  # Make sure ODS declaration and definitions are generated before compiling this.
  DEPENDS
  UkernelDialectIncGen

  # Link in the transform dialect, an all generated dialects.
  LINK_LIBS PRIVATE
  MLIRTransformDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRVectorDialect
)
