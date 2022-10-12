/* 
Script to generate custom rootfiles with the data format of Perkeo 3's measurement in 18/19.
Specificially for unittest (pedestal, detsum, ratedependency).
*/

#include <time.h>

#define NPMTDATA 1
#define DIFF 2000
#define NQDCS 8
#define TDCMAXSTAMPS 4
#define TDCDATASIZE (2 * (NQDCS + 1) * TDCMAXSTAMPS)

#define NUM_EVENTS 10000
#define CYCLENUMBER 650
#define CYCLESTART 12300

/*Define specific fill for PMTs for Pedestal(1), Detsum(2), Ratedependency(3)*/
#define TEST 3
#define INVALIDS 0

TRandom3 *RandomGenerator = new TRandom3(time(NULL));

/// Detector event
struct dataTreeData {
	UInt_t cycle; ///< Corresponding cycle number
	Double_t pmt[2 * NQDCS][NPMTDATA]; ///< All of the QDC values
	UChar_t detector; ///< Detector index
	UInt_t triggerTime=10000; ///< Time of trigger at Control card
	UInt_t deltaTriggerTime; ///< Time of trigger at Control card relative to last chopper pulse
	UInt_t eventNumber; ///< Unique event number in current cycle
	UInt_t cointime[2] = {69, 69};
};


/// Cycle data
struct cycleTreeData {
	UInt_t cycle; ///< Current cycle
	Bool_t valid; ///< Validity of cycle
	// UInt_t selectorSpeed; ///< Speed of selector
	Double_t chopperSpeed; ///< Measured chopper speed
	UInt_t realtime;
};


void define_dataTree_branches(TTree*, dataTreeData*);
void define_cycleTree_branches(TTree*, cycleTreeData*);

void fill_dataTree(TTree*, dataTreeData*, int event, Bool_t vaild, int cyc);
void fill_cycleTree(TTree*, cycleTreeData*, int event, Bool_t vaild, int cyc);


void fill_pmt_pedestal(dataTreeData*, Bool_t val, int num, int det);
void fill_pmt_detsum(dataTreeData*, Bool_t val);
void fill_pmt_ratedependency(dataTreeData*, Bool_t val, int num, int det);



void gen_root(){
	/* main function to create fill and save a rootfile */

	srand(time(NULL));

	dataTreeData *Data = new dataTreeData;
	cycleTreeData *Cycle = new cycleTreeData;
	
	TFile *output = new TFile("test.root", "RECREATE");
	TTree *dataTree = new TTree("dataTree", "dataTree");
	TTree *cycleTree = new TTree("cycleTree", "cycleTree");

	define_dataTree_branches(dataTree, Data);
	define_cycleTree_branches(cycleTree, Cycle);

	/* Main loop to fill NUM_EVENTS events */

	Bool_t valid = 1;
	int cyc = CYCLESTART;
	for (int event = 0; event < NUM_EVENTS; event++){
		if(INVALIDS){valid = rand() % 2;}
		else{valid = 1;}
		if(event % CYCLENUMBER == 0){ 
			fill_cycleTree(cycleTree, Cycle, event, valid, cyc);
			Data->triggerTime = 160000;
			cyc++;
		}
		fill_dataTree(dataTree, Data, event, valid, cyc);
	}
	
	output->Write();
	output->Close();
		
}



void define_dataTree_branches(TTree *dataTree, dataTreeData *Data){
	/**/
	dataTree->Branch("Cycle", &Data->cycle, "Cycle/i");
	if(NPMTDATA == 1){dataTree->Branch("PMT", &Data->pmt, "PMT[16][1]/D");}
	else if(NPMTDATA == 2){dataTree->Branch("PMT", &Data->pmt, "PMT[16][2]/D");}
	dataTree->Branch("Detector", &Data->detector, "Detector/b");
	dataTree->Branch("EventNumber", &Data->eventNumber, "EventNumber/i");
	dataTree->Branch("TriggerTime", &Data->triggerTime, "TriggerTime/i");
	dataTree->Branch("DeltaTriggerTime", &Data->deltaTriggerTime, "DeltaTiggerTime/i");
	dataTree->Branch("CoinTime", &Data->cointime, "CoinTime[2]/i");
	
}

void define_cycleTree_branches(TTree *cycleTree, cycleTreeData *Cycle){
	/**/
	cycleTree->Branch("Cycle", &Cycle->cycle, "Cycle/i");
	cycleTree->Branch("Valid", &Cycle->valid, "Valid/O");
	cycleTree->Branch("RealTime", &Cycle->realtime, "RealTime/i");
	cycleTree->Branch("ChopperSpeed", &Cycle->chopperSpeed, "ChopperSpeed/D");
	
}

void fill_cycleTree(TTree *cycleTree, cycleTreeData *Cycle, int event, Bool_t val, int cyc){
	/**/
	Cycle->cycle = cyc;
	Cycle->valid = val;
	Cycle->realtime =  RandomGenerator->Gaus(1001000000, 500000);
	cycleTree->Fill();
}

void fill_dataTree(TTree *dataTree, dataTreeData *Data, int event, Bool_t val, int cyc){
	/**/
	int det = rand() % 2;
	UInt_t trigtime = Data->triggerTime + 10000;
	if(TEST == 1){fill_pmt_pedestal(Data, val, event, det);}
	else if (TEST == 2){fill_pmt_detsum(Data, val);}
	else if (TEST == 3){fill_pmt_ratedependency(Data, val, event, det);}
	Data->detector = det;
	Data->eventNumber = event;
	Data->cycle = cyc;
	Data->triggerTime = trigtime;
	Data->deltaTriggerTime = 500000;
	dataTree->Fill();
	
}


/** CUSTOM PMT FILL FUNCTIONS FOR EACH UNITTEST -> fill_pmt_UNITTEST**/

void fill_pmt_pedestal(dataTreeData *Data, Bool_t val, int num, int det){
	/**/ 
	int det1, det2;
	int mean_sig, std_sig, mean_ped, std_ped;
	if (det  == 0){det1=0; det2=8;}
	else{det1=8; det2=0;}
	if(val == 1){
		mean_sig = 1000;
		std_sig = 50;
		mean_ped = 10;
		std_ped = 40;
	}
	else if(val == 0){
		mean_sig = 800;
		std_sig =  50;
		mean_ped = 40;
		std_ped =  10;
	}
	if (NPMTDATA == 1){
		for (int p=0; p<NQDCS; p++){
			Data->pmt[p+det1][0] = RandomGenerator->Gaus(mean_sig, std_sig);
			Data->pmt[p+det2][0] = RandomGenerator->Gaus(mean_ped, std_ped);
		}
	}
	else if (NPMTDATA == 2){
		for (int q=0; q<NQDCS; q++){
			Data->pmt[q+det1][0] = RandomGenerator->Gaus(DIFF, std_sig);
			Data->pmt[q+det2][0] = RandomGenerator->Gaus(DIFF, std_ped);
			Data->pmt[q+det1][1] = RandomGenerator->Gaus(mean_sig + DIFF, std_sig);
			Data->pmt[q+det2][1] = RandomGenerator->Gaus(mean_ped + DIFF, std_ped);
		}		
	}	
}


void fill_pmt_detsum(dataTreeData *Data, Bool_t val){
	/**/
	int mean_sig, std_sig; 
	if(val == 1){
		mean_sig = 1000;
		std_sig = 50;
	}
	else if(val == 0){
		mean_sig = 800;
		std_sig =  50;
	}
	if (NPMTDATA == 1){
		for (int p=0; p<2*NQDCS; p++){
			Data->pmt[p][0] = RandomGenerator->Gaus(mean_sig, std_sig);
		}
	}
	else if (NPMTDATA == 2){
		for (int q=0; q<2*NQDCS; q++){
			Data->pmt[q][0] = RandomGenerator->Gaus(DIFF, std_sig);
			Data->pmt[q][1] = RandomGenerator->Gaus(mean_sig + DIFF, std_sig);
		}		
	}	
}


void fill_pmt_ratedependency(dataTreeData *Data, Bool_t val, int num, int det){
	/**/ 
	int det1, det2;
	int mean_sig1, std_sig1, mean_sig2, std_sig2;
	if (det % 2 == 0){det1=0; det2=8;}
	else{det1=8; det2=0;}
	if(val == 1){
		mean_sig1 = 3000;
		std_sig1 = 50;
		mean_sig2 = 100;
		std_sig2 = 50;
	}
	else if(val == 0){
		mean_sig1 = 800;
		std_sig1 =  50;
		mean_sig2 = 150;
		std_sig2 =  50;
	}
	if (NPMTDATA == 1){
		for (int p=0; p<NQDCS; p++){
			if (num%2==0){
				Data->pmt[p+det1][0] = RandomGenerator->Gaus(mean_sig1, std_sig1);
				Data->pmt[p+det2][0] = RandomGenerator->Gaus(mean_sig1, std_sig2);
			}
			else{
				Data->pmt[p+det1][0] = RandomGenerator->Gaus(mean_sig2, std_sig2);
				Data->pmt[p+det2][0] = RandomGenerator->Gaus(mean_sig2, std_sig2);	
			}
		}
	}
	else if (NPMTDATA == 2){
		for (int q=0; q<NQDCS; q++){
			if (num%2==0){
				Data->pmt[q+det1][0] = RandomGenerator->Gaus(DIFF, std_sig1);
				Data->pmt[q+det2][0] = RandomGenerator->Gaus(DIFF, std_sig1);
				Data->pmt[q+det1][1] = RandomGenerator->Gaus(mean_sig1 + DIFF, std_sig1);
				Data->pmt[q+det2][1] = RandomGenerator->Gaus(mean_sig1 + DIFF, std_sig1);
			}
			else{
				Data->pmt[q+det1][0] = RandomGenerator->Gaus(DIFF, std_sig2);
				Data->pmt[q+det2][0] = RandomGenerator->Gaus(DIFF, std_sig2);
				Data->pmt[q+det1][1] = RandomGenerator->Gaus(mean_sig2 + DIFF, std_sig2);
				Data->pmt[q+det2][1] = RandomGenerator->Gaus(mean_sig2 + DIFF, std_sig2);
				
			}
		}		
	}	
}
