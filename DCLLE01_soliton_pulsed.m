clear
clc
close all

%4/26/16

c = 2.9979e8; % speed of light (m/s)

lc = 1550*1e-9; % laser wavelength (meters)

wl_lim=[1050 2450];
wl_tick=1050:400:2450;


%% define number of t points to step through and time grid
tphot_end=2000; % end time in terms of photon lifetimes
Nt=2e5;         % ???
h_phot=5e-3;    % ???
h=h_phot/2;     % ???
tol=5e-6;       % ???




%% for silica (silicon nitride??)
% user specified values
no=2; %index of refraction
fr=33e9; %rep rate
% Q=7.9e5; %105 fs, 200 mW
% Q=1.12e6; %94 fs, 100 mW
% Q=1.5e6; %100 fs, 56 mW
% Q=1.4e6; %94 fs, 64 mW
% Q=2e6;  %105 fs, 31 mW
% % Q=5e6; %272 fs, 5 mW
% % Q=3e6; %140 fs, 14 mW
% Q=2.885e5;
mode_area=(1e-6)^2;

n2=24e-20; %m^2/W (for silicon nitride)
hbar=1e-34;

% these values follow from those above
tRT=1/fr; %round trip time
diameter=tRT*(c/no)/pi; %resonator diameter, approximate
r=diameter/2;
fc=c/lc;
wc=2*pi*fc;

circumference=tRT*(c/no);
g0=n2*c*hbar*wc.^2/(no^2*circumference*mode_area)

%helpful information to display while code runs


%% define number of theta points to calculate (equivalently, number of
% modes carried in simulation)
Ntheta=2^nextpow2(16000);
dtheta=2*pi/Ntheta;


theta=(-Ntheta/2:Ntheta/2-1)*dtheta;
k=[0:Ntheta/2-1 -Ntheta/2:-1]; % because this is how Matlab orders its modes for the FFT
kfreq=fc+fr*k;
kwave=c./kfreq;



%% define parameters of physical system:
% alpha=41/30;
% rho=1.1;

% single soliton with beta(2) = -8e-
% alpha=2.0;
% rho=2.1;

% % nice soliton with beta(2) = -8e-5
alpha=2.5;
F2 = 3.30

% normal nice
% alpha=2.5;
% F2 = 10

% F2=(1+(rho-alpha)^2)*rho
% F=sqrt(F2)
    
dPM=7.5*pi;
% dPM=20*pi;

F=cos(pi/2*sin(theta/2).^2).*exp(1i*dPM*cos(theta));
Fch=F;
F=fftshift(ifft(abs(fft(F))));
F=F/max(abs(F))*sqrt(F2);


pump_power=5e-3

dwo=sqrt(pump_power/mean(abs(F).^2)*8*g0/(hbar*wc)) % ???
Q=wc/dwo % External Q-factor?
PP_no_pulse=max(abs(F).^2)*dwo^2*hbar*wc/(8*g0)




fh=figure;
DWb=1120*1e-9;
DWr=2*DWb;
DWbk=(c./(DWb)-c./lc)/fr;
DWrk=(c./(DWr)-c./lc)/fr;
Nc=(DWbk+DWrk)/2;
Ns=DWbk-Nc;

%%%%%%%%%%%%%%%%%%%%%%%%
beta(1)=-7.3e-4;

%beta(2)=-1e-5;
beta(2)=-8e-5;

beta(4)=12*beta(2)/(Nc^2-Ns^2);
beta(3)=-1*beta(4)*Nc/2;
beta(5)=1e-13;
beta(6)=6e-16;
beta(7)=-2.2e-18;

% beta(4)=0
% beta(3)=0
% beta(5)=0
% beta(6)=0
% beta(7)=0

fh2=figure
dalpha=0;
for j=1:length(beta)
    dalpha=dalpha+1i*beta(j)/factorial(j)*k.^j;
end
% NicePlot([kwave'*1e9,imag(dalpha')]);
M=sortrows([kwave'*1e9,imag(dalpha')],1);
area(M(:,1),M(:,2));
axis([wl_lim -100 100]);
set(gca,'xtick',wl_tick);

figure(fh);
subplot(2,1,1);
D=-beta*dwo/(4*pi);
D2=D(2:end);
temp=0;
for jj=1:length(D2)
    temp=temp+D2(jj)^jj.*k'.^(jj-1)/factorial(jj-1);
end
D2=temp*1e-3;

Dgvd=D2*1e3*no/c*1/fr^2*2*pi*c./kwave'.^2*1e12/(1e-3*1e9);
M=sortrows([kwave'*1e9,Dgvd],1);

[ax,h1,h2]=plotyy(kwave'*1e9,D2,M(:,1),M(:,2));
axis(ax(1),[wl_lim -320 200]);
axis(ax(2),[wl_lim -640 400]);
set(ax,'xtick',wl_tick);
set(ax,'linewidth',2,'fontname','calibri','fontsize',22);
set([h1 h2],'linewidth',2);
set(ax(1),'ytick',[-200 0 200]);
set(ax(2),'ytick',[-400 0 400]);
xlabel('\lambda (nm)');
ylabel(ax(1),'D_2 (kHz/mode)');
ylabel(ax(2),'D_{GVD} (ps/nm km)');

% pump_power_2=0.5;
% dwo_2=sqrt(pump_power_2/mean(abs(F).^2)*8*g0/(hbar*wc))
% Q_2=wc/dwo_2
% PP_no_pulse_2=max(abs(F).^2)*dwo_2^2*hbar*wc/(8*g0)
% D_2=-beta*dwo_2/(4*pi);
% D2=D_2(2:end);
% temp=0;
% for jj=1:length(D2)
%     temp=temp+D2(jj)^jj.*k'.^(jj-1)/factorial(jj-1);
% end
% D2=temp;
% 
% 
% hold on
% Dgvd_2=D2_2*1e3*no/c*1/fr^2*2*pi*c./kwave'.^2*1e12/(1e-3*1e9);
% M=sortrows([kwave'*1e9,0.5*Dgvd_2],1); % cut by 1/2 here because I have to plot it on the left axis, 
% %which has half the scale of the right axis
% h3=plot(ax(1),kwave'*1e9,D2_2,'linewidth',1,'color',get(h1,'color'),'linewidth',2,'linestyle','--');
% plot(ax(1),M(:,1),M(:,2),'linewidth',1,'color',get(h2,'color'),'linewidth',2,'linestyle','--');
% 
% hL=legend([h1 h3],['P=' num2str(pump_power*1e3) 'mW (' num2str(PP_no_pulse,1) 'W), Q=7e5'],['P=' num2str(pump_power_2*1e3) 'mW (' num2str(PP_no_pulse_2,2) 'W), Q=2e5'],'location','south');
% % set(hL,'units','normalized');
% % P=get(hL,'position');
% % P(1)=0.93*P(1);
% % set(hL,'position',P);
% legend boxoff



disp(['modal dispersion: ' num2str(-beta(2)*dwo/(2*2*pi)) ' Hz/mode']) %was -0.01 1:44 pm 6/25/15
mod_freq_diff=-dwo*beta(1)/(2*2*pi)





disp(['photon lifetime is ' num2str(1/dwo*1e9) ' nanoseconds']);
disp(['resonance linewidth is ' num2str(dwo/(2*pi*1e6)) ' MHz']);

figure(fh2);
xlabel('\lambda (nm)');
ylabel('\Delta\alpha(\lambda)');
set(gca,'fontsize',22,'fontname','calibri','linewidth',2);
axis([1050 2250 -12 12]);



%% make linear operator for dispersion, detuning, and loss
D=-(1+1i*alpha);
for j=1:length(beta)
    D=D+1i*beta(j)/factorial(j)*k.^j;
end

%% define pumping waveform




figure(fh);
subplot(2,1,2);
h1=NicePlot(sortrows([kwave'*1e9,10*log10(abs(fft(F')/max(abs(fft(F)))).^2)],1),'o');
color1=get(h1,'color');
axis([1525 1575 -60 5]);
xlabel('\lambda (nm)');
ylabel('P (dB)');
text(1545,-35,{'f_{rep}=33 GHz','PM depth 22.5\pi'},'fontname','calibri','fontsize',22,'color',color1);
P=get(gcf,'position');
P(2)=P(2)/2;
P(4)=1.3*P(4);
set(gcf,'position',P);
set(gca,'units','inches');
set(ax,'units','inches');
P(3)=1.3*P(3);
set(gcf,'position',P);
P=get(gca,'position');
P(1)=P(1)+0.5;
P(2)=P(2)+0.15;
set(gca,'position',P);
set(gca,'units','normalized');
P=get(ax(1),'position');
P(1)=P(1)+0.5;
P(2)=P(2)+0.18;
set(ax,'position',P);
set(ax,'units','normalized')
F=F.*exp(1i*cos(theta)*10*pi);

P=get(gcf,'position');
P([3 4])=P([3 4])*1.2;
set(gcf,'position',P);




%% define input waveform





A=zeros(size(theta)); %psi is the single-time wavefunction of theta
A_in=A; %save it for later


%% iteration scheme
dtphot_plot=1; %plot every dtphot_plot photon lifetimes
iter=0;
piter=1;

I=abs(A).^2; %intensity


runningfig=figure;

%intensity part of plot
subplot(3,1,1);
hp=NicePlot([theta'/pi,I']);
color1=get(hp,'color');
xlabel('\theta/\pi ');
ylabel('I (arb)');
axis([-1 1 0 3]);
hold on
hp1_2=NicePlot([theta'/pi,abs(F').^2/max(abs(F).^2)*max(I)]);
color2=get(hp1_2,'color');
NicePlot([theta'/pi,abs(Fch').^2/max(abs(Fch).^2)]);

%spectrum part of plot
subplot(3,1,2);
spec=abs(fft(A)).^2;
spec=10*log10(spec/max(spec)); %normaize and convert to dB
hs=bar(kwave*1e9,spec,'basevalue',-100,'linewidth',1,'edgecolor',color1,'facecolor',color1);
% set(hs,'edgealpha',0.1);
set(gca,'fontsize',22,'fontname','calibri','linewidth',2);
xlabel('\lambda (nm)');
ylabel('Power (dB)');
axis([wl_lim -100 0]);
set(gca,'xtick',wl_tick);
drawnow
hold on

specin=10*log10(abs(fft(F)).^2); specin=specin-max(specin);
bar(kwave*1e9,specin,'basevalue',-100,'linewidth',1,'edgecolor',color2,'facecolor',color2);

subplot(3,1,3);
hp2=NicePlot([theta'/pi*tRT*1e15,I']);
xlabel('t (fs)');
ylabel('I (arb)');
axis([-1500 1500 0 3]);
hold on
hp3=NicePlot([theta'/pi*tRT*1e15,abs(F').^2/max(abs(F).^2)*max(I)]);

Tfs=theta'/pi*tRT*1e15;

min_field=sqrt(2*g0/dwo*1/2); %half a photon per mode

%%do the iteration

tphot=0;
tphot_plot=dtphot_plot;

pump_adj=0;


Do=D;


while tphot<tphot_end
    
%     if tphot>20
%         [~,ind]=max(I);
%         tmax=theta(ind)'/pi*tRT*1e15;
%         D=Do+1i*1e-4*tmax*k;
%     end



    iter=iter+1;
    error=1;
    
    h=2*h; %to account for the halving on the first iteration
    Aft=fft(A);
    
    while error>2*tol
        %% take one step
        
        AI=ifft(exp(D*h/2).*Aft);
        k1=ifft(exp(D*h/2).*fft(h*(1i*abs(A).^2.*A+F)));
        
        Ak2=AI+k1/2;
        k2=h*(1i*abs(Ak2).^2.*Ak2+F);
        
        Ak3=AI+k2/2;
        k3=h*(1i*abs(Ak3).^2.*Ak3+F);
        
        Ak4=ifft(exp(D*h/2).*fft(AI+k3));
        k4=h*(1i*abs(Ak4).^2.*Ak4+F);
        
        A1=ifft(exp(D*h/2).*fft(AI+k1/6+k2/3+k3/3))+k4/6;
        
        %% take two half steps
        h=h/2;
        A2=A;
        for jstep=1:2
            AI=ifft(exp(D*h/2).*Aft);
            k1=ifft(exp(D*h/2).*fft(h*(1i*abs(A2).^2.*A2+F)));
            
            Ak2=AI+k1/2;
            k2=h*(1i*abs(Ak2).^2.*Ak2+F);
            
            Ak3=AI+k2/2;
            k3=h*(1i*abs(Ak3).^2.*Ak3+F);
            
            Ak4=ifft(exp(D*h/2).*fft(AI+k3));
            k4=h*(1i*abs(Ak4).^2.*Ak4+F);
            
            A2=ifft(exp(D*h/2).*fft(AI+k1/6+k2/3+k3/3))+k4/6;
            if jstep>1
                break
            end
            Aft=fft(A2);
        end
        
        %% get the error
        
%         Af=4/3*A2-1/3*A1;
        error=sqrt(sum(abs(A2-A1).^2)/sum(abs(A2).^2));
        
    end
    h_phot=4*h;
    tphot=tphot+h_phot;
    Aft=fft(4/3*A2-1/3*A1);
    Aft=Aft+min_field*exp(1i*2*pi*rand(size(A))).*(abs(Aft)<min_field);
    A=ifft(Aft);
    I=abs(A).^2;
    
    h=2^(-1/3)*h*(error>tol)+2^(1/3)*h*(error<=tol);
    
    
    
    
    
    
    
    drawnow
    %runnning data storage and plotting
    if tphot>tphot_plot;
        Alog(piter,:)=A;
        Aftlog(piter,:)=Aft;
        tphotlog(piter)=tphot;
        subplot(3,1,1);
        set(hp,'xdata',theta'/pi,'ydata',I');
        set(hp1_2,'ydata',abs(F').^2/max(abs(F).^2)*max(I));
        title(['iter ' num2str(iter) ', ' num2str(tphot,3) '\tau_\gamma']);
%                 title([num2str(tSI(piter+1)*1e6,2) ' \mus, iter ' num2str(iter)]);
%                 set(ht,'string',[num2str(tPhot(piter+1),2) '\tau_{phot}, ' num2str(mean(I),5) ' avg pow' ]);
%                 set(ht,'string',[num2str(tPhot(piter+1),2) '\tau_{phot}']);
%                 set(htext2,'string',['RTs=' num2str(round(RTs(q)))]);
        subplot(3,1,2);
        spec=10*log10(abs(Aft).^2);
        spec=spec-max(spec)-pump_adj*(k~=0);
        
       
        
        
        set(hs,'ydata',spec);
        
         subplot(3,1,1);
        set(hp2,'ydata',I');
        set(hp3,'ydata',abs(F').^2/max(abs(F).^2)*max(I));
        drawnow
        
        piter=piter+1;
        tphot_plot=tphot_plot+dtphot_plot;
    end
end
return
subplot(3,1,1);
title('');

subplot(3,1,3);
firstind=find(I>max(I)/2,1,'first');
lastind=find(I>max(I)/2,1,'last');
t_first=Tfs(firstind);
t_last=Tfs(lastind);
FWHM=t_last-t_first;
text(500,1.2,['FWHM ' num2str(FWHM,2) 'fs'],'fontsize',22,'fontname','calibri','color',color1);
% text(-1400,1.2,{['pump P= ' num2str(pump_power*1e3,2) 'mW'],...
%     ['Q=' num2str(Q,6)]},'fontsize',22,'fontname','calibri','color',color1);

set(gcf,'units','normalized');
set(gcf,'position',[0.3 0.15 0.4 0.7]);
drawnow
set(gcf,'paperpositionmode','auto');
print('final output three DWs', '-dpng', '-r400');

figure(fh);
set(gcf,'paperpositionmode','auto');
print('input and dispersion three DWs','-dpng','-r400');

figure(fh2);
set(gcf,'paperpositionmode','auto');
xlabel('\lambda (nm)');
ylabel('\alpha(\lambda)');
print('alpha lambda three DWs','-dpng','-r400');




