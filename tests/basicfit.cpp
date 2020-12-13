

int basicfit(const char* Sfile, const char* Outfile, int hpar0 = 10, int hpar1 = 0, int hpar2 = 10, int fit_limlow = 0., int fit_limup = 10.){

    TH1D* hist1 = new TH1D("hist1", "hist1", hpar0, hpar1, hpar2);
    hist1->Sumw2();
    TH1D* hist2 = new TH1D("hist2", "hist2", hpar0, hpar1, hpar2);
    hist2->Sumw2();
    int x;
    ifstream inFile;

    inFile.open(Sfile);
    if (!inFile) {
        cout << "Unable to open file";
        exit(1); // terminate with error
    }
    while (inFile >> x) {
        hist1->Fill(x);
        hist2->Fill(x+2);
    }
    inFile.close();

    hist2->Scale(-0.5);
    hist1->Add(hist2);

    hist1->Fit("pol0", "", "", fit_limlow, fit_limup);
    TF1* fitS = hist1->GetFunction("pol0");

    float p0 = fitS->GetParameter(0);
    float p0_err = fitS->GetParError(0);
    float chi2 = fitS->GetChisquare();
    float ndf = fitS->GetNDF();

    //Output
    hist1->Print("all");
    std::cout << p0 << "   " << p0_err << "   " << float(chi2)/float(ndf) << std::endl;

    ofstream myfile;
    myfile.open (Outfile);
    myfile << std::setprecision(9) << p0 << "\t" << p0_err << "\t" << float(chi2)/float(ndf) << "\n";
    myfile.close();

    gApplication->Terminate();
    return 0;
}