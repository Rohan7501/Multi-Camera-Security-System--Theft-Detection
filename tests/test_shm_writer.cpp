#include<sys/mman.h>
#include<fcntl.h>
#include<unistd.h>
#include<cstdio>

int main(){
    const char* name = "/demo_int";
    int fd  = shm_open(name, O_CREAT | O_RDWR, 0666);
    ftruncate(fd, sizeof(int));
    int *p = (int*)mmap(nullptr, sizeof(int), PROT_READ|PROT_WRITE, MAP_SHARED, fd,0);
    *p = 513;
    printf("wrote %d to %s\n", *p, name);
    munmap(p,sizeof(int));
    close(fd); 

    return 0;
}
