hf download --repo-type dataset umkc-mcc/nga --local-dir /scratch/zb7df/data/nga
unzip -q /scratch/zb7df/data/nga/train.zip -d /scratch/zb7df/data/nga
rm /scratch/zb7df/data/nga/train.zip
unzip -q /scratch/zb7df/data/nga/validation.zip -d /scratch/zb7df/data/nga
rm /scratch/zb7df/data/nga/validation.zip
unzip -q /scratch/zb7df/data/nga/test_256.zip -d /scratch/zb7df/data/nga
rm /scratch/zb7df/data/nga/test_256.zip

hf download --repo-type dataset umkc-mcc/sandia --local-dir /scratch/zb7df/data/sandia
unzip -q /scratch/zb7df/data/sandia/train.zip -d /scratch/zb7df/data/sandia
rm /scratch/zb7df/data/sandia/train.zip
unzip -q /scratch/zb7df/data/sandia/validation.zip -d /scratch/zb7df/data/sandia
rm /scratch/zb7df/data/sandia/validation.zip
unzip -q /scratch/zb7df/data/sandia/test_256.zip -d /scratch/zb7df/data/sandia
rm /scratch/zb7df/data/sandia/test_256.zip

hf download zb-umkc/sar-pact --local-dir /scratch/zb7df/models/sar-pact