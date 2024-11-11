`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: 
// Engineer: 
// 
// Create Date: 18.02.2024 15:21:28
// Design Name: 
// Module Name: float_adder
// Project Name: 
// Target Devices: 
// Tool Versions: 
// Description: 
// 
// Dependencies: 
// 
// Revision:
// Revision 0.01 - File Created
// Additional Comments:
// 
//////////////////////////////////////////////////////////////////////////////////


module float_adder #(
        parameter EXP_WIDTH = 8,
        parameter SFD_WIDTH = 7
    ) (
        input [EXP_WIDTH+SFD_WIDTH:0] a,
        input [EXP_WIDTH+SFD_WIDTH:0] b,
        output [EXP_WIDTH+SFD_WIDTH:0] out
    );
    
    wire a_sign = a[EXP_WIDTH+SFD_WIDTH];
    wire [EXP_WIDTH-1:0] a_exp = a[EXP_WIDTH+SFD_WIDTH-1:SFD_WIDTH];
    wire [SFD_WIDTH-1:0] a_sfd = a[SFD_WIDTH-1:0];
    
    wire b_sign = b[EXP_WIDTH+SFD_WIDTH];
    wire [EXP_WIDTH-1:0] b_exp = b[EXP_WIDTH+SFD_WIDTH-1:SFD_WIDTH];
    wire [SFD_WIDTH-1:0] b_sfd = b[SFD_WIDTH-1:0];
    
    reg out_sign;
    reg [EXP_WIDTH-1:0] out_exp;
    reg [SFD_WIDTH-1:0] out_sfd;
    
    assign out = {out_sign, out_exp, out_sfd};
    
    
    wire a_l = {a_exp, a_sfd} >= {b_exp, b_sfd}; //sorting into large and small by magnitude (l and s)
    wire l_sign = a_l ? a_sign : b_sign;
    wire [EXP_WIDTH-1:0] l_exp = a_l ? a_exp : b_exp;
    wire [SFD_WIDTH-1:0] l_sfd = a_l ? a_sfd : b_sfd;
    wire s_sign = a_l ? b_sign : a_sign;
    wire [EXP_WIDTH-1:0] s_exp = a_l ? b_exp : a_exp;
    wire [SFD_WIDTH-1:0] s_sfd = a_l ? b_sfd : a_sfd;
        
    
    wire l_exc = &l_exp; //inputs exceptional (inf or nan)
    wire s_exc = &s_exp;
    wire l_sub = ~(|l_exp); //inputs subnormal
    wire s_sub = ~(|s_exp);
    
    wire [EXP_WIDTH-1:0] l_exp_true = l_exp | l_sub; //true exponents (corrected for subnormal)
    wire [EXP_WIDTH-1:0] s_exp_true = s_exp | s_sub;
    
    wire [EXP_WIDTH-1:0] exp_diff = l_exp_true - s_exp_true; //difference between exponents
    wire subtract = l_sign ^ s_sign; //subtract if signs different
    
    wire [SFD_WIDTH+4:0] l_fsfd = {1'b0, ~l_sub, l_sfd, 3'b0}; //full significand for l
    
    //full significand for s - account for subtraction and shift and sticky bit
    wire [SFD_WIDTH+1:0] s_isfd1 = {1'b0, ~s_sub, s_sfd}; //significand with explicit leading bit and carry-out bit
    wire [SFD_WIDTH+4:0] s_isfd2 = {subtract ? -s_isfd1 : s_isfd1, 3'b0}; //invert for subtraction if needed, make space for guard+round+sticky
    wire [SFD_WIDTH+4:0] s_stickies;
    assign s_stickies[0] = 1'b0; //sticky zero in case of no shift
    generate genvar i; //generate stickies for each shift value
        for (i = 1; i <= SFD_WIDTH+4; i++) assign s_stickies[i] = s_stickies[i-1] | s_isfd2[i];
    endgenerate
    wire [SFD_WIDTH+4:0] s_fsfd = {$signed(s_isfd2) >>> exp_diff} | (exp_diff > SFD_WIDTH+4 ? s_stickies[SFD_WIDTH+4] : s_stickies[exp_diff]);
    
    wire [SFD_WIDTH+4:0] fsfd_sum = l_fsfd + s_fsfd; //sum of full significands
    reg [$clog2(SFD_WIDTH+3)-1:0] sum_zero_count;
    reg [$clog2(SFD_WIDTH+3)-1:0] shift_amt;
    reg [SFD_WIDTH+2:0] shifted_sum_sfd; //two bits shorter: no carry out bit, no guard bit (processed into round and sticky)
    reg [EXP_WIDTH-1:0] new_exp;
    
    //nearest-rounded significand sum: previously calculated sum, +1 if (lowest bit & round bit) or (round bit & sticky bit)
    wire [SFD_WIDTH+1:0] rounded_sum_sfd = {1'b0, shifted_sum_sfd[SFD_WIDTH+2:2]} + (&(shifted_sum_sfd[2:1]) | &(shifted_sum_sfd[1:0]));
    
    always @* begin
        //sum normalization handling
        if (fsfd_sum[SFD_WIDTH+4]) begin //uppermost bit: shift 1 right, increase exponent
            sum_zero_count = 'b0; //(doesn't matter)
            shift_amt = 'b0; //(doesn't matter)
            shifted_sum_sfd = {fsfd_sum[SFD_WIDTH+4:3], fsfd_sum[2] | fsfd_sum[1] | fsfd_sum[0]}; //shift 1 right, maintaining sticky bit
            new_exp = l_exp_true + 1;
        end else if (fsfd_sum[SFD_WIDTH+3]) begin //already normalized: no shifting needed
            sum_zero_count = 'b0; //(doesn't matter)
            shift_amt = 'b0; //(doesn't matter)
            shifted_sum_sfd = {fsfd_sum[SFD_WIDTH+3:2], fsfd_sum[1] | fsfd_sum[0]}; //no shift, but consolidate guard+round+sticky into round+sticky
            new_exp = l_exp_true;
        end else if (~(|fsfd_sum)) begin //zero: result is zero
            sum_zero_count = 'b0; //(doesn't matter)
            shift_amt = 'b0; //(doesn't matter)
            shifted_sum_sfd = 'b0;
            new_exp = 'b0;
        end else begin //shift left to normalize
            for (sum_zero_count = 1; sum_zero_count < SFD_WIDTH+3; sum_zero_count++) begin //count zeros (except for last, else we'd be in zero case)
                if (fsfd_sum[SFD_WIDTH+3-sum_zero_count]) break; //skip width+4 (uppermost bit), must be zero else we'd be in first case
            end
            if (sum_zero_count >= l_exp_true) begin //normalizing would underflow exponent: shift as far as possible and denormalize
                shift_amt = l_exp_true - 1;
                new_exp = 'b0;
            end else begin //exponent is fine: shift until normalized
                shift_amt = sum_zero_count;
                new_exp = l_exp_true - sum_zero_count;
            end
            shifted_sum_sfd[SFD_WIDTH+2:1] = fsfd_sum[SFD_WIDTH+3:2] << shift_amt; //shift (extra bit from guard - overwritten next line)
            shifted_sum_sfd[1:0] = shift_amt == 'd1 ? fsfd_sum[1:0] : 2'b0; //retain or clear round+sticky bits
        end
    
        //output
        if (l_exc || s_exc) begin //infinities/nans
            if ((l_exc && |l_sfd) || (s_exc && |s_sfd)) begin //at least one is nan: return nan
                out_sign = 'b0;
                out_exp = {EXP_WIDTH{1'b1}};
                out_sfd = {{(SFD_WIDTH-1){1'b0}}, 1'b1};
            end else begin //at least one is infinity
                if (s_exc && (l_sign != s_sign)) begin //both are different infinities: return nan
                    out_sign = 'b0;
                    out_exp = {EXP_WIDTH{1'b1}};
                    out_sfd = {{(SFD_WIDTH-1){1'b0}}, 1'b1};
                end else begin //both same infinity, or only l is infinity: return that infinity
                    out_sign = l_sign;
                    out_exp = {EXP_WIDTH{1'b1}};
                    out_sfd = 'b0;
                end
            end
        end else begin //only actual numbers
            out_sign = l_sign;
            if (&new_exp) begin //overflow during addition itself: infinity
                out_exp = {EXP_WIDTH{1'b1}};
                out_sfd = 'b0;
            end else if (rounded_sum_sfd[SFD_WIDTH+1]) begin //carry-out during rounding: shift right, increase exponent
                if (&(new_exp[EXP_WIDTH-1:1])) begin //increasing exponent would overflow: return infinity
                    out_exp = {EXP_WIDTH{1'b1}};
                    out_sfd = 'b0;
                end else begin //increasing exponent is fine: return shifted/increased results
                    out_exp = new_exp + 1;
                    out_sfd = rounded_sum_sfd[SFD_WIDTH:1];
                end
            end else begin //general case: no overflows or shifts needed, return result
                out_exp = new_exp;
                out_sfd = rounded_sum_sfd[SFD_WIDTH-1:0];
            end
        end
    end
    
endmodule
