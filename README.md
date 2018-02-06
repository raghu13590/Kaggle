# Kaggle

The Project aims to classify genetic variations from clinical literature.

Data is stored in two separate files - training_text(columns separated by '||') and training_variants (',' delimited). 
After adding both the files into a dataframe, the data would look like this,

                                                 Text    Gene             Variation  Class
ID                                                                                        
0   Cyclin-dependent kinases (CDKs) regulate a var...  FAM58A  Truncating Mutations      1
1    Abstract Background  Non-small cell lung canc...     CBL                 W802*      2
2    Abstract Background  Non-small cell lung canc...     CBL                 Q249E      2
3   Recent evidence has demonstrated that acquired...     CBL                 N454D      3
4   Oncogenic mutations in the monomeric Casitas B...     CBL                 L399V      4
5   Oncogenic mutations in the monomeric Casitas B...     CBL                 V391I      4
6   Oncogenic mutations in the monomeric Casitas B...     CBL                 V430M      5
7   CBL is a negative regulator of activated recep...     CBL              Deletion      1
8    Abstract Juvenile myelomonocytic leukemia (JM...     CBL                 Y371H      4
9    Abstract Juvenile myelomonocytic leukemia (JM...     CBL                 C384R      4
10  Oncogenic mutations in the monomeric Casitas B...     CBL                 P395A      4
11  Noonan syndrome is an autosomal dominant conge...     CBL                 K382E      4
12  Noonan syndrome is an autosomal dominant conge...     CBL                 R420Q      4
13  Noonan syndrome is an autosomal dominant conge...     CBL                 C381A      4
14  Oncogenic mutations in the monomeric Casitas B...     CBL                 P428L      5
15  Noonan syndrome is an autosomal dominant conge...     CBL                 D390Y      4
16  To determine if residual cylindrical refractiv...     CBL  Truncating Mutations      1
17  Acquired uniparental disomy (aUPD) is a common...     CBL                 Q367P      4
18  Oncogenic mutations in the monomeric Casitas B...     CBL                 M374V      5
19  Acquired uniparental disomy (aUPD) is a common...     CBL                 Y371S      4

Text - clinical evidence to classify genetic mutation
Gene - the gene where this genetic mutation is location
Variation - aminoacid change for the mutation
Class (dependent variable) - One of 9 classes the genetic mutation has been classified on.

The idea is to break the 'Text' column into muliple columns filtering out common english words using NLP techniques and running various classification algortihms.

