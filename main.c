#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

#define MAX_ITERATIONS 0xFFFFFFFF

//directive <pragma omp>

/*
    blocco di test calcolo pi, Leibniz
    for (int j=0; j<100; j++){
            if (j%2)
                somm_ex=somm_ex+(float)((-1)/(float)((2*j)+1));
            else
                somm_ex=somm_ex+(float)((1)/(float)((2*j)+1));
            printf("%f\n", (4.0*somm_ex));
    }

 */


int approx_pi(){

    long long int j=0; //serve per il contatore dato che eccede i 4byte
    int thr_id, max_thr=omp_get_max_threads();// la prima usata per salvare privatamente gli id di ogni thread, la seconda per capire di quanti thread dispongo
    float  sum=0; //salvo il risultato finale della reduction
    double start, end, partial_start, partial_end; //tempi di esecuzione

    start=omp_get_wtime();
#pragma omp parallel default(none) shared(j, sum, max_thr) private(thr_id, partial_start, partial_end) //definisco default(none) in modo che il programma non assegni arbitrariamente private o shared alle variabili
    {
        thr_id=omp_get_thread_num();
        partial_start=omp_get_wtime();

#pragma omp for reduction(+ : sum) //approssimazione di pi usando leibniz, separando i casi di (-1)^n portando la complessità da n^2 a n
        for (j=0; j<MAX_ITERATIONS; j++){
            if (j%2)
                sum=sum+((-1)/(float)((2*j)+1));
            else
                sum=sum+((1)/(float)((2*j)+1));

            if(!(j%1000)&&thr_id==0){//j%1000 per non sprecare troppe risorse a calcolare il tempo. le operazioni dentro questo if verranno eseguite solo dal thread num 0
                partial_end=omp_get_wtime();
                if((partial_end-partial_start)>0.2){
                    printf("\r0%.1f\%\n", (float)((((j*1000)/MAX_ITERATIONS)/10)*max_thr));

                    /* confronto j con il massimo di iterazioni così da ottenere la percentuale di progressione.
                       j*1000 perchè senza probabilmente si va incontro a qualche fenomeno di cancellazione numerica
                       dato che solo il thread 0 sta operando vuol dire che lavora su un j che va da 0 a 2^32/max_thread, quindi devo tenerne conto moltiplicando la percentuale di avanzamento per il numero totale di thread
                       questa operazione non influisce sul risultato, dato che essendo un'esecuzione in parallelo ogni thread si trova circa allo stesso punto in ogni momento
                    */
                    partial_start=partial_end;
                }

            }
        }


    }
    end=omp_get_wtime();
    printf("%.21f\n", (4.0*sum));
    printf("iterazioni eseguite in %.10f secondi usando %d threads\n\n", (end-start), max_thr);


}



void setup(void){
    // nulla di particolare, permette di eseguire le iterazioni impostando manualmente il numero di threads
    int num_thr;
    printf("inserire quanti thread usare per l'esecuzione (al massimo 8), 0 per uscire:\n");
    scanf("%d", &num_thr);
    if (num_thr<1||num_thr>8)
        exit(0);
    omp_set_num_threads(num_thr);
    printf("esecuzione con %d threads\n", omp_get_max_threads());
}



int main(int argc, char** argv) {
    printf("Inizio test da 1 a 8 core\nAlla fine del test si potranno selezionare manualmente i threads\nverranno eseguite %ux iterazioni sulla formula di Leibniz\n\n", MAX_ITERATIONS);
    int counter =1;
    while (counter<9){
        omp_set_num_threads(counter);
        approx_pi();
        counter++;
    }
    while(1){
        setup();
        approx_pi();

    }

}
