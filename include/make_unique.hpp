#ifndef MAKE_UNIQUE
#define MAKE_UNIQUE

/*
 * This is the version of make_unique written by Stephan T. Lavavej who
 * originally proposed adding this function to C++14. This is from stack
 * overflow and is consistent with the N3656 standard revision.
 * 
 * https://stackoverflow.com/questions/17902405/how-to-implement-make-unique-function-in-c11/17902439#17902439
 *
 * This is in the reply from sasha.sochka
 */

#if __cplusplus < 201402L

#include <memory>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace std
{

  template<class T>
  struct _Unique_if
  {
    typedef std::unique_ptr<T> _Single_object;
  };

  template<class T> struct
  _Unique_if<T[]>
  {
    typedef unique_ptr<T[]> _Unknown_bound;
  };

  template<class T, size_t N>
  struct _Unique_if<T[N]>
  {
    typedef void _Known_bound;
  };

  template<class T, class... Args>
  typename _Unique_if<T>::_Single_object make_unique(Args&&... args)
  {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
  }

  template<class T>
  typename _Unique_if<T>::_Unknown_bound make_unique(size_t n)
  {
    typedef typename remove_extent<T>::type U;
    return unique_ptr<T>(new U[n]());
  }

  template<class T, class... Args> typename _Unique_if<T>::_Known_bound make_unique(Args&&...) = delete;

}

#endif

#endif
