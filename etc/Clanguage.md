## 1차원 배열

- 1차원 배열 선언은 아래와 같이 배열 요소의 자료형/ 배열 이름/ 배열 길이 를 포함함.

```cpp
int oneDimArr[4];
```

- 배열 위치 인덱스 값은 1이 아니라 0부터 시작함.
<br>

- 배열 길이보다 초기화 값 수가 작으면 나머지는 0으로 초기화됨.

```cpp
int arr3[5] = {1,2}
// arr3은 {1,2,0,0,0}
```

- 문자열을 배열로 선언할 수 있음. 단, 맨 마지막 글자는 null문자로 고정됨.
  - null문자는 문자열의 끝을 구분하기 위한 도구임.
  
```cpp
char str[] = "Good morning!"
// 배열의 길이는 null문자를 포함해 14가 됨
```

- 문자열을 배열로 선언 시 바로 문자열을 입력하지 않고 선언만 할 경우, 크기를 지정해 줘야 함.

```cpp
char str[50]; // char str[] 로는 쓸 수 없음, 즉 문자열을 바로 입력하지 않고 선언만 하려면 크기를 설정해야 함
```

- 문자열을 scanf로 입력받을 때는 변수명 앞에 &를 붙이지 않음.
  - 후술하겠지만, 배열 이름을 \[ \] 없이 쓰면 배열의 첫 번째 요소의 주소로 decay되기 때문.

```cpp
char str[50];
scanf("%s", str); // char 자료형으로 선언된 배열에 대해서는 &str 이 아님
```

- 위 코드에서 공백을 포함하는 문장 입력 시 공백 기준으로 잘림. 연산자를 변경하면 문장 입력 가능.

```cpp
char str[50];
scanf("%[^\n]s", str); // \n을 입력하기 전까지는 (즉 엔터 치기 전까지는) 계속 문자열을 받아들임
```

## 포인터의 이해

- 포인터 변수는 메모리의 주소 값을 저장하기 위한 변수임.
<br>

- 포인터 변수에는 특정 변수의 주소 값을 저장할 수 있음.

```cpp
int num = 7;
int * pnum; // int형 변수의 주소 값을 저장하기 위한 포인터 변수 pnum 선언
pnum = &num; // num의 주소 값을 포인터 변수 pnum에 저장
```

- pnum에는 int형 변수 num (4바이트) 의 시작 번지 주소 값이 저장됨. 이 때 포인터 변수 pnum이 int형 변수 num을 가리킨다고 함. 
  - num이 메모리에서 0x12ff76~0x12ff79 의 4개 바이트를 차지하고 있다면, pnum=0x12ff76 임.
  - 64비트 시스템에서는 1개 포인터 변수가 8바이트를 차지함 (8바이트 x 8비트/바이트 = 64비트)
    <br><br>

- 아래 예제에서는 두 포인터 변수들 ptr1과 ptr2 모두 정수 변수 num을 가리킴.

```cpp
int num = 10;
int * ptr1 = &num; // ptr1에는 num의 시작 번지 주소 값이 저장됨
int * ptr2 = ptr1; // ptr2에는 ptr1의 시작 번지 주소가 아니라 ptr1의 값, 즉 num의 시작 번지 주소 값이 저장됨
```

- 기 선언된 포인터 변수의 앞에 \*를 붙인 후 값을 대입하면, 그 포인터 변수가 가리키는 변수에 값을 대입함.

```cpp
int num = 10;
int * pnum = &num; // 포인터 변수 pnum의 선언, num을 가리키도록 함
*pnum = 20; // pnum이 가리키는 대상 (pnum 자신이 아님) 에 20 대입, num=20 과 동일한 결과를 얻음
printf("%d\n",*pnum); // num의 값 20이 출력됨 (pnum의 값이 아님에 주의)
printf("%p\n",pnum); // 포인터 변수 pnum의 값 (서식문자가 %p이므로, num의 값이 저장된 메모리 공간의 주소)
```

- 포인터는 선언 후 타 변수의 주소값 (&변수명) 혹은 0 또는 NULL로 초기화해야 함.

```cpp
int * ptr; // 포인터가 어디를 가리키는 지 모름, 이를 쓰레기 값으로 초기화되었다고 함
*ptr = 200; // 포인터가 가리키는 곳에 200 대입, 만약 포인터가 가리키는 위치가 중요한 곳이라면 치명적일 수 있음
// 선언만 해 두고 값은 나중에 할당할 거라면, int * ptr = NULL; 로 초기화해줄 것
```

## 포인터와 배열

- 배열의 이름은 마치 포인터처럼, 배열의 첫 요소의 주소를 가리키도록 decay됨.

```cpp
int arr[3] = {0,1,2};
int * ptr = arr; // 배열 이름이 첫 요소의 주소를 가리키도록 decay되므로, 포인터 변수 선언에 배열 이름을 쓸 수 있음
printf("%p\n", arr); // 배열의 이름을 인덱스 없이 그대로 출력하면 주소가 출력됨 (이를테면 0012FF50)
// %p는 주소 값 출력에 사용되는 서식문자
printf("%p\n", &arr[0]); // 0012FF50 -> 첫 요소의 주소 (&를 붙였으므로), arr 출력 결과와 동일
printf("%p\n", &arr[1]); // 0012FF54 -> int 크기가 4바이트이므로 4 증가
printf("%p\n", &arr[2]); // 0012FF58 -> int 크기가 4바이트이므로 4 증가
printf("%p\n", ptr); // 0012FF50 ->  배열 첫 요소의 주소가 출력됨
```

- 배열 이름은 (포인터 변수와 달리) 대입 연산자의 피연산자가 될 수 없음 (즉 값을 변경할 수 없음).
<br>

- 배열 이름 앞에 \*를 붙여 연산 수행 자체는 가능 (배열 이름이 배열의 첫 요소의 주소를 가리키므로).

```cpp
int arr1[3] = {1, 2, 3};
*arr1 += 1; // arr1은 {2, 2, 3} 이 됨. *arr1이 arr1[0]의 주소를 가리키므로
```

- 포인터 변수에 직접 1을 더하면, 해당 자료형의 크기만큼 주소 값이 증가함.
  - 이를테면 int형 포인터의 경우 주소 값이 4 증가함
    <br><br>
    
- 포인터 변수에 직접 숫자를 더하는 것으로 배열 접근이 가능함, arr[i] 는 \*(arr+i)와 같음.

```cpp
int arr[3] = {1, 2, 3};
int * ptr = arr; // int * ptr = &arr[0] 과 같음
printf("%d %d %d", *ptr, *(ptr+1), *(ptr+2); // 각각 arr[0], arr[1], arr[2]와 같음
// 이는 (ptr+1)에 의해 주소 값이 4 증가했고, 정수 배열에 대해 주소 값 4 증가는 다음 정수 요소를 의미하기 때문
```

- 문자열 선언은 아래와 같이 포인터 기반으로 할 수도 있음. 이 경우 포인터 변수는 첫 번째 char형 문자의 주소 값을 저장함.
  - 포인터 기반 문자열 선언도 문자열 선언이므로, 포인터 변수를 서식문자 %s로 printf 출력 시 문자열을 그대로 출력함
  
```cpp
char * str = "Your string"; // 특정 메모리 공간에 문자열이 저장되고 (맨 끝에 null 문자 포함), str에는 Y의 주소 값이 저장됨
// char * str; 처럼 선언만 하고 문자열을 입력하지 않으면 warning이 뜸
printf("%s\n", str); // 서식문자가 %s이므로 Your string 이 출력됨
```

- 배열로 선언된 문자열은 그 내용을 변경할 수 있으나, 포인터로 선언된 문자열은 그 내용을 변경할 수 없음.
  - 포인터로 선언된 문자열을 '상수 형태의 문자열' 이라고도 함.
  
```cpp
char str1[] = "My string";
char * str2 = "Your string";
str1[0] = 'X'; // 문자열 변경 가능
str2[0] = 'X'; // 오류 발생함, 문자열 변경 불가능
*str2[0] = 'X'; // 오류 발생함, str2는 포인터 변수이지 배열이 아니기 때문
*str2 = 'X'; // 오류 발생함, 포인터로 선언된 문자열은 그 내용을 변경할 수 없음
```

- 포인터 배열 (각 요소가 포인터 변수인 배열) 로 문자열의 배열을 선언할 수 있음.

```cpp
char * strArr[3] = {"Simple", "String", "Array"}; // 배열의 각 원소는 각 문자열이 (정확히는 각 문자열의 첫 번째 문자가) 차지하는 메모리 공간의 주소
printf("%s %s %s", strArr[0], strArr[1], strArr[2]); // 서식문자가 %s이므로 각각 Simple, String, Array가 출력됨
```

## 포인터와 함수

- 배열을 함수의 인자로 전달하려면, 함수 선언 시 매개변수를 포인터 변수로 선언해야 함. 

```cpp
void SimpleFunc(int * param) // 정수 포인터 변수 param이 정수 배열을 의미 (int * ptr = arr; 이 가능했던 걸 상기)
{ 
    printf("%d \n", param[1]); // 전달받은 정수 1차원 배열의 두 번째 요소 출력
}

int main(void)
{
    int arr[3] = {1, 2, 3};
    SimpleFunc(arr); // 정수 배열 이름을 그대로 사용, 두 번째 요소의 값인 2가 출력됨
    return 0;
}
```

- 위 코드에서 배열을 함수의 인자로 전달받기 위한 함수 선언 시의 매개변수 선언을 아래와 같이 할 수도 있음.
  - 배열을 인자로 전달한다는 느낌을 더 직관적으로 주므로, 아래 형태 선언을 사용하기도 함.
  - 함수의 매개변수로써 배열을 의미하는 포인터 변수 선언 시, 크기는 입력하지 않음.
  
```cpp
void SimpleFunc(int param[]) // void SimpleFunc(int * param) 과 같음
// 단, 매개변수가 아닌 일반적인 포인터 변수 선언에서 위와 같이 대체하지는 못함
// 이를테면 int * ptr = arr; 을 int ptr[] = arr; 로 쓸 수 없음
```

- 문자열은 배열이므로, 함수가 문자열을 인자로 전달받기 위해서도 매개변수를 포인터 변수로 선언해야 함.

```cpp
void SimpleFunc(char * sentence) // 문자열을 인자로 받기 위한 매개변수는 char 포인터 변수
{ 
    printf("%s \n", sentence); 
}
int main(void)
{
    char sentence1[] = "My string";
    SimpleFunc(sentence1); // My string 출력
    char * sentence2 = "Your string";
    SimpleFunc(sentence2); // Your string 출력
    SimpleFunc("Our string"); // Our string 출력
    return 0;
}
```

- 함수호출 시 전달되는 인자의 값은 매개변수에 '복사' 됨. 그러므로 두 변수의 값을 서로 바꾸는 함수 구현은 아래와 같이 할 수 없음.

```cpp
void Swap(int n1, int n2) // n1과 n2는 num1과 num2의 값을 복사받을 뿐, num1과 num2 자체가 아님
{
    int temp = n1;
    n1 = n2;
    n2 = temp;
}
int main(void)
{
    int num1 = 10;
    int num2 = 20;
    Swap(num1, num2); // num1과 num2는 바뀌지 않음
    return 0;
}
```

- 두 변수의 값을 바꾸려면, 함수의 매개변수를 포인터 변수로 두고 두 정수 변수의 주소 값을 전달해 정수 변수들에 접근할 수 있도록 해야 함.

```cpp
void Swap(int * ptr1, int * ptr2) // 포인터 변수 ptr1에는 num1의 주소가 입력됨, 즉 ptr1이 num1을 가리킴
{
    int temp = *ptr1; // ptr1이 가리키는 변수 (num1)의 값이 temp에 입력됨
    *ptr1 = *ptr2; // ptr1이 가리키는 변수 (num1) 에 ptr2가 가리키는 변수 (num2) 의 값이 입력됨
    *ptr2 = temp; // ptr2가 가리키는 변수 (num2) 에 temp 값 (즉 기존 num1의 값) 이 입력됨
}
int main(void)
{
    int num1 = 10;
    int num2 = 20;
    Swap(&num1, &num2); // 주소를 전달함(& 연산자), num1과 num2의 값이 서로 바뀜
    return 0;
}
```

- 정수를 입력받기 위해 scanf 함수 호출 시 정수 변수 이름 앞에 & 연산자를 붙이는 이유는, 해당 정수 변수의 주소를 알아야 해당 변수에 접근해서 값을 채울 수 있기 때문임.

## 구조체

- 구조체 포인터 변수도 선언할 수 있음. 구조체 포인터 변수를 이용해 구조체의 멤버변수에 접근하고 연산할 수 있음.

```cpp
struct point
{
    int xpos;
    int ypos;
}; // 구조체 정의문에서는 마지막에 ;를 붙여야 함
int main(void)
{
    struct point pos1 = {1, 2}; // 구조체 변수 정의
    struct point * pptr = &pos1; // point 구조체 포인터 변수 pptr이 구조체 pos1을 가리킴
    
    pos1.xpos += 1;
    pos1.ypos += 2;
    (*pptr).xpos += 3; // *pptr은 pptr이 가리키는 대상이므로 pos1임
    (*pptr).ypos += 4;
    pptr -> xpos += 5; // (*pptr).xpos += 5; 와 동일함 (-> 연산자를 쓰는 경우도 많으므로 익숙해질 것)
    pptr -> ypos += 6;
    
    return 0;
}
```

- typedef 선언을 통해, 구조체 정의 부분 외 나머지 부분에서 struct를 생략할 수 있음.

```cpp
typedef struct point
{
    int xpos;
    int ypos;
} Point; // struct point에 Point라는 이름 부여

int main(void)
{
    Point pos1 = {1, 2}; // struct point 대신 Point 사용 가능
    Point * pptr = &pos1;   
    pptr -> xpos += 5;
    pptr -> ypos += 6;
    printf("%d %d \n", pos1.xpos, pos1.ypos);
    return 0;
}
```

- 구조체를 함수에 전달하고 함수를 통해 구조체의 멤버변수에 접근하려면, 매개변수로 구조체 포인터 변수를 선언해야 함

```cpp
typedef struct point
{
    int xpos;
    int ypos;
} Point; // struct point에 Point라는 이름 부여

void OrgSymTrans(Point * ptr) // Point 구조체를 가리키는 포인터 매개변수
{
    ptr->xpos = (ptr->xpos) * -1; // ptr->xpos 는 (*ptr).xpos 와 동일
    ptr->ypos = (ptr->ypos) * -1;
}

int main(void)
{
    Point pos1 = {1, 2};    
    OrgSymTrans(&pos1); // & 붙여서 구조체 변수의 주소 전달 (함수의 매개변수가 포인터 변수이므로)
    return 0;
}
```


## 메모리 관리와 메모리 동적 할당

- 아래 코드를 실행하면, 지역변수 name의 '주소' 를 반환함 (char 배열로 선언되었으므로 배열 이름은 주소를 담음), 그런데 지역변수 문자열은 ReadUserName 함수를 빠져나오면서 소멸됨, 그러므로 printf의 결과가 (NULL)이 됨.

```cpp
#include <stdio.h>

char * ReadUserName(void)
{
    char name[30]; // 함수 내에서 지역적으로 선언됨, 즉 ReadUserName 함수가 종료되면 소멸됨
    printf("What's your name? ");
    gets(name);
    return name; // 포인터로 선언된 문자열을 반환하므로, 결과적으로 name의 '주소'를 반환함
}

int main(void)
{
    char * name1;
    char * name2;
    
    name1 = ReadUserName();
    printf("name1: %s \n", name1); // (NULL)이 뜸, name1이 가리키는 주소의 문자열은 ReadUserName이 종료되면서 소멸되었기 때문
    name2 = ReadUserName();
    printf("name2: %s \n", name2);
    
    return 0;
}
```

- malloc 함수를 쓰면, 힙 영역에 메모리 공간을 할당해 함수가 종료되어도 값이 유지되도록 함.

```cpp
#include <stdio.h>
#include <stdlib.h> // malloc, free 함수 쓰기 위해 필요

char * ReadUserName(void)
{
    char * name = (char *)malloc(sizeof(char)*30); // malloc 함수로 힙에 공간 할당
    printf("What's your name? ");
    gets(name);
    return name;
}

int main(void)
{
    char * name1;
    char * name2;
    
    name1 = ReadUserName();
    printf("name1: %s \n", name1); // 입력한 문자열이 정상적으로 출력됨
    name2 = ReadUserName();
    printf("name2: %s \n", name2);
    
    free(name1); // 힙에 할당된 공간 해제
    free(name2);
    return 0;
}
```

- 동적 할당의 예시로, 특별히 종료하지 않는 한 계속 인적사항을 입력받아 추가하는 직원 관리 프로그램을 들 수 있음 (입력 화면에서 직원 한 명의 인적사항 입력 가능, 엔터 누르면 다시 다른 직원의 인적사항 입력하는 화면이 뜸, 원하는 경우에 입력 중단 및 종료 가능).<br> 
몇 명의 인적사항을 입력할 지 미리 정해주는 것이 아닌 한, 메모리가 얼마나 필요할 지 모름 (서구권처럼 사람 이름 길이가 천차만별이면 더더욱). 그러므로 미리 메모리를 확보해놓는 것이 아니라, 인적사항을 입력받을 때마다 (즉 '동적으로') 힙에 메모리 공간을 할당함.

## C++에서의 참조자

- 새로 선언되는 변수 이름 앞에 &를 붙이면, 참조자의 선언을 뜻함. 참조자는 해당 메모리 공간에 또 다른 이름을 붙이는 것임.

```cpp
int num1 = 10;
int *ptr = &num1; // &가 이미 선언된 변수 이름 앞에 오므로 주소 값 반환임
int &num2 = num1; // &가 새로 선언되는 변수 이름 앞에 오므로 참조자임

cout << num1 << endl;
cout << num2 << endl; // 두 출력문은 동일하게 10을 출력함
```

- 참조자의 수에는 제한이 없으며, 참조자를 대상으로 참조자를 선언할 수도 있음.
<br>

- 참조자는 변수에 대해서만 선언이 가능하며, 선언됨과 동시에 무언가를 참조해야 함.
  - 즉 아래의 코드를 컴파일하면 에러 발생.

```cpp
int &ref1 = 20; // 참조자를 상수 대상으로 선언할 수 없어 에러
int &ref2; // 참조자를 선언하면서 참조 대상을 명시하지 않아 에러
```

- 배열의 요소와 포인터 변수에 대해 참조자 선언이 가능함 (배열 자체에 대해서는 아님).

```cpp
int num = 10;
int *ptr = &num;
int *(&pref) = ptr; // pref는 포인터 변수 ptr의 참조자 (포인터 변수 ptr에 pref라는 또 다른 이름 부여)
cout << *pref << endl; // num의 값 10 출력
```

- 함수의 매개변수 선언 시 참조자를 사용하면, 현재 함수의 외부에서 선언된 변수에 접근할 수 있음.
  - 참조자를 이용하면 포인터 설명에 사용한 Swap 예제를 구현할 수 있음.

```cpp
void Swap(int &ref1, int &ref2) // 참조자 선언으로, num1과 num2의 메모리 공간에 ref1과 ref2라는 이름이 추가로 붙음
{
    int temp = ref1; // 참조자를 통해 num1과 num2의 메모리 공간에 접근 가능함 (num1과 num2는 Swap함수 외부에서 초기화되었음에도)
    ref1 = ref2;
    ref2 = temp;
}
int main(void)
{
    int num1 = 10;
    int num2 = 20;
    Swap(num1, num2); // 주소를 전달하지 않고 num1과 num2를 그대로 넣어도, num1과 num2의 값이 서로 바뀜
    return 0;
} 
```

- 참조자를 함수의 매개변수로 사용하면, 새로운 변수를 선언해 메모리 공간을 확보하고 그 공간에 매개변수 값을 복사해 초기화할 필요 없이 기존 메모리 공간을 참조만 하면 되므로 성능상 이점이 있음.

```cpp
int sum(int &a, int &b) 
// 만약 매개변수를 참조자 없이 int a, int b로 선언할 경우 a와 b를 위한 메모리 공간을 새로 확보하고 num1과 num2의 값을 붙여넣어야 해 비효율적임. 그러나 참조자를 쓰면 num1과 num2를 참조하기만 하면 되어 효율적임.
{
	return (a + b);
}
int main()
{
	int num1 = 10, num2 = 20;
	int result = sum(num1, num2);
	return 0;
}
출처: https://koey.tistory.com/42 [황민혁의 기술 블로그:티스토리]

```

- 위에서 Swap함수의 매개변수로 참조자를 쓰면 Swap 함수에 &num1과 &num2가 아닌 num1과 num2를 그대로 입력할 수 있는데, 이는 코드 검토의 측면에서는 단점이 되기도 함. 
  
```cpp
void SomeFunc(int &ref) // 참조자 선언으로, num의 메모리 공간에 ref라는 이름이 추가로 붙음
{
    (특정 명령들) // ref를 이용해 num의 값을 바꾸는 내용 (이를테면 ref++) 일 수도 있고, 아닐 수도 있음
}
int main(void)
{
    int num = 10;
    SomeFunc(num);
    cout << num << endl; // 문제점: SomeFunc의 내용을 보지 않고서는, 10이 출력될지 다른 숫자가 출력될지 알 수 없음!
    // 만약 C언어였다면 당연히 10이 출력될 것 (C언어에는 사용자 정의 함수의 매개변수를 참조자로 선언하는 기능이 없으므로)
    // 이러한 단점 때문에, 참조자 사용을 싫어하는 프로그래머도 적지 않다고 함
    return 0;
}
```

- 위 단점을 보완하기 위해, 함수 내에서 참조자를 통한 외부 변수의 값 변경이 진행되지 않도록 하는 장치인 const 선언을 할 수 있음.

```cpp
void SomeFunc(const int &ref) // 위 예시에서 const를 참조자 매개변수 선언 앞에 붙이면, SomeFunc에서 num의 값을 변경할 수 없음
// const를 붙여서 선언할 경우, SomeFunc에서 ref++; 작성 시 에러 뜸 (SomeFunc 내에서 num의 값을 변경할 수 없는데 변경하려고 했으므로) 
```

- 참조자를 함수의 매개변수로 선언하면 해당 함수에는 상수를 입력할 수 없음 (참조자가 상수를 참조할 수 없으므로). 그러나 const 참조자를 함수 매개변수로 선언하면, 상수 입력이 가능함.

```cpp
void SomeFunc(const int &ref) // const 참조자 선언
{
    (특정 명령들) // ref를 이용해 num의 값을 바꾸는 내용 (이를테면 ref++) 일 수도 있고, 아닐 수도 있음
}
int main(void)
{
    SomeFunc(10); // SomeFunc에서 매개변수를 const 참조자로 선언하면 에러 없이 작동함, 그러나 const를 없애면 에러 발생
    return 0;
}
```

- 클래스의 멤버함수에서 반환형을 해당 클래스에 대한 참조자로, 반환대상을 \*this 로 하면 해당 클래스 객체에 대해 해당 함수를 chaining 형식으로 (Python에서 함수들을 이어붙여 사용하듯이) 쓸 수 있음.
  - this는 객체 자신을 가리키는 포인터임, 그러므로 \*this는 자기 자신임.

```cpp
class SelfRef
{
private:
    int num;
public:
    SelfRef(int n) : num(n) {}
    SelfRef& Adder(int n) // SelfRef 뒤에 &를 붙였으므로, 반환형이 SelfRef 객체에 대한 참조자
    {
        num += n;
        return *this; // 객체 자신을 가리키는 포인터가 this이므로, *this는 객체 자신
        // 즉 SelfRef 객체 자신에 대한 참조자가 반환됨
    }
    SelfRef& ShowNumber() // 반환형이 SelfRef 객체에 대한 참조자
    {
        cout<<num<<endl;
        return *this; // 객체 자신을 가리키는 포인터가 this이므로, *this는 객체 자신
        // 즉 SelfRef 객체 자신에 대한 참조자가 반환됨
    }
};
int main(void)
{
    SelfRef obj(3); // SelfRef 객체 선언
    SelfRef &ref = obj; // SelfRef 객체에 대한 참조자 선언
    ref.Adder(1).ShowNumber(); // 함수들을 chaining 형식으로 사용 (사실 ref 대신 obj 써도 됨)
    return 0;
}        
```

# TODOS

- strcpy 정리 필요

- C++ 컴파일러 중 strcpy를 쓰면 에러가 나는 경우가 있음. 이 때는 아래 매크로를 코드 최상단에 추가해야 함.
```cpp
#define _CRT_SECURE_NO_WARNINGS
```


```python

```
