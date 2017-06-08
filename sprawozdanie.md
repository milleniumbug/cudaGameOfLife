Wykorzystane optymalizacje:
- wykonywanie operacji na wielu strumieniach
- asynchroniczne kopiowanie
- oddzielenie na fazę obliczeń i fazę kopiowań
- zwiększenie pamięci podręcznej: cudaDeviceSetCacheConfig(cudaFuncCachePreferL1)

GTX 650, 16x16 wątków, 2000 generacji, plansza (5x512)x(5x512) z optymalizacjami:
Executed in: 30285932 microseconds (30285 milliseconds)
GTX 650, 32x32 wątków, 2000 generacji, plansza (5x512)x(5x512) z optymalizacjami:
Executed in: 27666721 microseconds (27666 milliseconds)
GTX 650, 16x16 wątków, 2000 generacji, plansza (10x512)x(10x512) z optymalizacjami:
Executed in: 82641518 microseconds (82641 milliseconds)
GTX 650, 32x32 wątków, 2000 generacji, plansza (10x512)x(10x512) z optymalizacjami:
Executed in: 79816204 microseconds (79816 milliseconds)
GTX 650, 16x16 wątków, 2000 generacji, plansza (5x512)x(5x512) bez optymalizacji:
Executed in: 45089952 microseconds (45089 milliseconds)
GTX 650, 32x32 wątków, 2000 generacji, plansza (5x512)x(5x512) bez optymalizacji:
Executed in: 43196536 microseconds (43196 milliseconds)
GTX 650, 16x16 wątków, 2000 generacji, plansza (10x512)x(10x512) bez optymalizacji:
Executed in: 138431606 microseconds (138 seconds)
GTX 650, 32x32 wątków, 2000 generacji, plansza (10x512)x(10x512) bez optymalizacji:
Executed in: 131996511 microseconds (131 seconds)

Tesla C2050, 16x16 wątków, 2000 generacji, plansza (5x512)x(5x512) z optymalizacjami:
Executed in: 12171490 microseconds (12171 milliseconds)
Tesla C2050, 32x32 wątków, 2000 generacji, plansza (5x512)x(5x512) z optymalizacjami:
Executed in: 13049188 microseconds (13049 milliseconds)
Tesla C2050, 16x16 wątków, 2000 generacji, plansza (10x512)x(10x512) z optymalizacjami:
Executed in: 36782640 microseconds (36782 milliseconds)
Tesla C2050, 32x32 wątków, 2000 generacji, plansza (10x512)x(10x512) z optymalizacjami:
Executed in: 39576129 microseconds (39576 milliseconds)
Tesla C2050, 16x16 wątków, 2000 generacji, plansza (5x512)x(5x512) bez optymalizacji:
Executed in: 13310863 microseconds (13310 milliseconds)
Tesla C2050, 32x32 wątków, 2000 generacji, plansza (5x512)x(5x512) bez optymalizacji:
Executed in: 14332918 microseconds (14332 milliseconds)
Tesla C2050, 16x16 wątków, 2000 generacji, plansza (10x512)x(10x512) bez optymalizacji:
Executed in: 40511882 microseconds (40511 milliseconds)
Tesla C2050, 32x32 wątków, 2000 generacji, plansza (10x512)x(10x512) bez optymalizacji:
Executed in: 43962410 microseconds (43962 milliseconds)



z cudaDeviceSetCacheConfig(cudaFuncCachePreferL1):
N 2000
Executed in: 301934519 microseconds (301 seconds)
z cudaDeviceSetCacheConfig(cudaFuncCachePreferL1):
N 2000
Executed in: 308640607 microseconds (308 seconds)
