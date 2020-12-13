

int filter(const char* Sfile, const char* Outfile, int hpar0 = 10, int hpar1 = 0, int hpar2 = 10, bool bfilter = 0){

    //Import file
	TFile* fsourceS = new TFile(Sfile);
	if (!fsourceS->IsOpen()) {
		cerr << "Error: Cannot open file " << Sfile << endl;
	}
	std::cout << Sfile << std::endl;

    TH1D* hist1 = new TH1D("hist1", "hist1", hpar0, hpar1, hpar2);
    hist1->Sumw2();

    TTree* dataTree = (TTree*) fsourceS->Get("dataTree");

    const char* pmt0 = "(PMT[0][1] - PMT[0][0])>>hist1";
    if (bfilter){
        const char* filter = "DeltaTriggerTime >= 380e3 && DeltaTriggerTime < 650e3";
        dataTree->Draw(pmt0, filter);
    }
    else{
        dataTree->Draw(pmt0);
    }

    hist1->Print("all");
    //hist1->Draw();

    ofstream myfile;
    myfile.open (Outfile);
    for(int i = 1; i < hpar0 +1; ++i){
        myfile << std::setprecision(9) <<  hist1->GetBinCenter(i) << "\t" << hist1->GetBinContent(i) << "\t" << hist1->GetBinError(i) << "\n";
    }
    myfile.close();

    gApplication->Terminate();
    return 0;
}