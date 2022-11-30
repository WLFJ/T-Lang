#include <iostream>
#include <cstdlib>
using namespace std;
/*
 *
 * def main(){
 *   a = [[1, 2]];
 *   b = [[1], [3]];
 *   c = [[100]];
 *   print(a . b . c); # matmul
 * }
 *
 */
int main(void){
  float* t = (float*)malloc(sizeof(float) * 1);
  float* a = (float*)malloc(sizeof(float) * 2);
  float* b = (float*)malloc(sizeof(float) * 2);
  float* c = (float*)malloc(sizeof(float) * 1);

  t[0] = 0.;
  a[0] = 1.;
  a[1] = 2.;
  b[0] = 1.;
  b[1] = 3.;
  c[0] = 100.;

  // i = 1, j = 2, k = 1
  for(int i = 0; i < 1; i ++){
    for(int j = 0; j < 2; j ++){
      for(int k = 0; k < 1; k ++){
        t[i + k] += a[i + j] * b[j + k];
      }
    }
  }

  for(int i = 0; i < 1; i ++){
    for(int j = 0; j < 1; j ++){
      for(int k = 0; k < 1; k ++){
        t[i + k] = t[i + j] * c[j + k];
      }
    }
  }

  cout << t[0 + 0] << endl;

  free(c);
  free(b);
  free(a);
  free(t);
  return 0;
  
}
