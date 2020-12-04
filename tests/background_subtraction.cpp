

int background_subtraction(const char* Sfile, int fitlim_low = 45e3, int fitlim_high = 52e3){

    //Import file
	TFile* fsourceS = new TFile(Sfile);
	if (!fsourceS->IsOpen()) {
		cerr << "Error: Cannot open file " << Sfile << endl;
	}
	std::cout << Sfile << std::endl;

    TTree* dataTree = (TTree*) fsourceS->Get("dataTree");

    //Detector 0

    TH1D* histSig = new TH1D("histSig", "histSig", 2000, 0, 52e3);
    TH1D* histBg = new TH1D("histBg", "histSig", 2000, 0, 52e3);
    histSig->Sumw2();
    histBg->Sumw2();

    const char* detsumS = "(PMT[0][1] - PMT[0][0] + PMT[1][1] - PMT[1][0] + PMT[2][1] - PMT[2][0] + PMT[3][1] - PMT[3][0] + PMT[4][1] - PMT[4][0] + PMT[5][1] - PMT[5][0] + PMT[6][1] - PMT[6][0] + PMT[7][1] - PMT[7][0])>>histSig";
    const char* detsumB = "(PMT[0][1] - PMT[0][0] + PMT[1][1] - PMT[1][0] + PMT[2][1] - PMT[2][0] + PMT[3][1] - PMT[3][0] + PMT[4][1] - PMT[4][0] + PMT[5][1] - PMT[5][0] + PMT[6][1] - PMT[6][0] + PMT[7][1] - PMT[7][0])>>histBg";

    dataTree->Draw(detsumS, "DeltaTriggerTime >= 380e3 && DeltaTriggerTime < 650e3");
    dataTree->Draw(detsumB, "DeltaTriggerTime >= 1150e3 && DeltaTriggerTime < 1200e3");
    histBg->Scale(-5.4);
    histSig->Add(histBg);

    //Detector 1

    TH1D* histSig1 = new TH1D("histSig1", "histSig1", 2000, 0, 52e3);
    TH1D* histBg1 = new TH1D("histBg1", "histSig1", 2000, 0, 52e3);
    histSig1->Sumw2();
    histBg1->Sumw2();

    const char* detsumS1 = "(PMT[8][1] - PMT[8][0] + PMT[9][1] - PMT[9][0] + PMT[10][1] - PMT[10][0] + PMT[11][1] - PMT[11][0] + PMT[12][1] - PMT[12][0] + PMT[13][1] - PMT[13][0] + PMT[14][1] - PMT[14][0] + PMT[15][1] - PMT[15][0])>>histSig1";
    const char* detsumB1 = "(PMT[8][1] - PMT[8][0] + PMT[9][1] - PMT[9][0] + PMT[10][1] - PMT[10][0] + PMT[11][1] - PMT[11][0] + PMT[12][1] - PMT[12][0] + PMT[13][1] - PMT[13][0] + PMT[14][1] - PMT[14][0] + PMT[15][1] - PMT[15][0])>>histBg1";

    dataTree->Draw(detsumS1, "DeltaTriggerTime >= 380e3 && DeltaTriggerTime < 650e3");
    dataTree->Draw(detsumB1, "DeltaTriggerTime >= 1150e3 && DeltaTriggerTime < 1200e3");
    histBg1->Scale(-5.4);
    histSig1->Add(histBg1);

    // Fitting

    histSig->Fit("pol0", "", "", fitlim_low, fitlim_high);
    TF1* fitS = histSig->GetFunction("pol0");
    histSig1->Fit("pol0", "", "", fitlim_low, fitlim_high);
    TF1* fitS1 = histSig1->GetFunction("pol0");

    float p0 = fitS->GetParameter(0);
    float p0_err = fitS->GetParError(0);
    float chi2 = fitS->GetChisquare();
    float ndf = fitS->GetNDF();

    float p01 = fitS1->GetParameter(0);
    float p0_err1 = fitS1->GetParError(0);
    float chi21 = fitS1->GetChisquare();
    float ndf1 = fitS1->GetNDF();

    //Output

    std::cout << p0 << "   " << p0_err << "   " << float(chi2)/float(ndf) << std::endl;
    std::cout << p01 << "   " << p0_err1 << "   " << float(chi21)/float(ndf1) << std::endl;

    ofstream myfile;
    myfile.open ("root_fitres.txt");
    myfile << p0 << "\t" << p0_err << "\t" << float(chi2)/float(ndf) << "\n";
    myfile << p01 << "\t" << p0_err1 << "\t" << float(chi21)/float(ndf1) << "\n";
    myfile.close();

    gApplication->Terminate();
    return 0;
}

