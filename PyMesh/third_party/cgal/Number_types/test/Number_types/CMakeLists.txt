# based on a script created by cgal_create_cmake_script
# TODO: currently, everything depends on CORE, it would be nice to break
# that dependency so as to test all the number types not depending on CORE
# when it is not installed

cmake_minimum_required(VERSION 3.1...3.15)
project( Number_types_Tests )



find_package( CGAL QUIET COMPONENTS Core )

if ( CGAL_FOUND )

  include(${CGAL_USE_FILE})
  include( CGAL_VersionUtils )

  include_directories( BEFORE include )

  create_single_source_cgal_program( "bench_interval.cpp" )
  create_single_source_cgal_program( "constant.cpp" )
  create_single_source_cgal_program( "CORE_BigFloat.cpp" )
  create_single_source_cgal_program( "CORE_BigInt.cpp" )
  create_single_source_cgal_program( "CORE_BigRat.cpp" )
  create_single_source_cgal_program( "CORE_Expr.cpp" )
  create_single_source_cgal_program( "Counted_number.cpp" )
  create_single_source_cgal_program( "double.cpp" )
  create_single_source_cgal_program( "doubletst.cpp" )
  create_single_source_cgal_program( "eigen.cpp" )
  create_single_source_cgal_program( "float.cpp" )
  create_single_source_cgal_program( "floattst.cpp" )
  create_single_source_cgal_program( "Gmpfr.cpp" )
  create_single_source_cgal_program( "Gmpq.cpp" )
  create_single_source_cgal_program( "Gmpq_new.cpp" )
  create_single_source_cgal_program( "Gmpz.cpp" )
  create_single_source_cgal_program( "Gmpzf_new.cpp" )
  create_single_source_cgal_program( "int.cpp" )
  create_single_source_cgal_program( "Interval_nt.cpp" )
  create_single_source_cgal_program( "Interval_nt_new.cpp" )
  create_single_source_cgal_program( "ioformat.cpp" )
  create_single_source_cgal_program( "known_bit_size_integers.cpp" )
  create_single_source_cgal_program( "Lazy_exact_nt.cpp" )
  create_single_source_cgal_program( "Lazy_exact_nt_new.cpp" )
  create_single_source_cgal_program( "leda_bigfloat.cpp" )
  create_single_source_cgal_program( "leda_bigfloat_interval.cpp" )
  create_single_source_cgal_program( "leda_integer.cpp" )
  create_single_source_cgal_program( "leda_rational.cpp" )
  create_single_source_cgal_program( "leda_real.cpp" )
  create_single_source_cgal_program( "long_double.cpp" )
  create_single_source_cgal_program( "long_doubletst.cpp" )
  create_single_source_cgal_program( "long_long.cpp" )
  create_single_source_cgal_program( "MP_Float.cpp" )
  create_single_source_cgal_program( "MP_Float_exact.cpp" )
  create_single_source_cgal_program( "mpq_class.cpp" )
  create_single_source_cgal_program( "mpz_class.cpp" )
  create_single_source_cgal_program( "Mpzf.cpp" )
  create_single_source_cgal_program( "Nef_polynomial.cpp" )
  create_single_source_cgal_program( "NT_checker.cpp" )
  create_single_source_cgal_program( "Quotient.cpp" )
  create_single_source_cgal_program( "quotient_io.cpp" )
  create_single_source_cgal_program( "root_of_2.cpp" )
  create_single_source_cgal_program( "Root_of_traits.cpp" )
  create_single_source_cgal_program( "rounding_modes.cpp" )
  create_single_source_cgal_program( "simplest_rational.cpp" )
  create_single_source_cgal_program( "Sqrt_extension_false.cpp" )
  create_single_source_cgal_program( "Sqrt_extension_true.cpp" )
  create_single_source_cgal_program( "_test_valid_finite_double.cpp" )
  create_single_source_cgal_program( "_test_valid_finite_float.cpp" )
  create_single_source_cgal_program( "to_interval_test.cpp" )
  create_single_source_cgal_program( "unsigned.cpp" )
  create_single_source_cgal_program( "utilities.cpp" )
  if( GMP_FOUND )
    find_package( MPFI )
    if( MPFI_FOUND )
      include( ${MPFI_USE_FILE} )
    endif( MPFI_FOUND )
  endif( GMP_FOUND )
  # all the programs below will be linked against MPFI in case it is present
  create_single_source_cgal_program( "Gmpfi.cpp" )
  create_single_source_cgal_program( "Gmpfr_bug.cpp" )
  create_single_source_cgal_program( "Quotient_new.cpp" )
  create_single_source_cgal_program( "test_nt_Coercion_traits.cpp" )

else( CGAL_FOUND )

  message(STATUS
    "These tests require the CGAL library and will not be compiled.")

endif( CGAL_FOUND )
